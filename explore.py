# -*- coding: utf-8 -*-
"""
pydex - Python Data Exploration Tool - Explore

@date: 2018-01-25

@author: luc.vandeputte@arcelormittal.com
"""

import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText
import tkinter.filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import pandas as pd
import datetime as datetime
import getpass
import graph as graph
import settings
import re
import tkSimpleDialog as tkSD	# for dialog boxes in menus
import traceback as tb        	# for showing callback windows at exceptions
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg # for interactive menu

class explore(ttk.Frame):
    def __init__(self, master = None, dataframes = None):
        # Construct the Frame object.
        ttk.Frame.__init__(self, master)
        # Bind click event
        self.bind_all('<Button-1>', self.click)
        # Place the main frame on the grid
        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)
        #self.pack()
		 # Create Menubar
        self.menubar = tk.Menu(self)
        # Create widgets
        self.createWidgets()
        # Init setting module with global variables
        if not hasattr(settings, 'dataframes'):
            settings.init()
        # Add dataframes passed as parameter to explorer's global dictionary
        if dataframes:
            for name, dataframe in dataframes.items():
                # Correct name of columns - replace spaces, etc
                dataframe.rename(columns=lambda x: ''.join(c for c in x if c not in ('()<>[]{},./?;:''"\|-=+*&^%$#@!`~')).strip().replace(' ', '_'), inplace=True)
                settings.dataframes[name] = dataframe
        # Add missing dataframes from dictionary of global variables to explorer's global dictionary
        for name, variable in globals().items():
            if isinstance(eval(name), pd.core.frame.DataFrame):
                # Correct name of columns - replace spaces, etc
                variable.rename(columns=lambda x: ''.join(c for c in x if c not in ('()<>[]{},./?;:''"\|-=+*&^%$#@!`~')).strip().replace(' ', '_'), inplace=True)
                settings.dataframes[name] = variable
        #set bins default as auto
        self.entryBins.insert(0, 'auto')
        # Do not wrap columns when display stats
        pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.expand_frame_repr', False)
		# Setup custom exception handling
        tk.Tk.report_callback_exception=self.HandleExceptionCommand
#        # Setup custom exception handling
#        self.report_callback_exception=self.handle_exception
#
#    # Callback function - Handle exceptions
#    def handle_exception(self, exception, value, traceback):
#        messagebox.showinfo('Error',value)
        
    # Callback function - Clicked somewhere
    def click(self, event):
        try:
            if self.focus_get() == self.entryY:
                self.selectedVar.set('Y')
            elif self.focus_get() == self.entryX:
                self.selectedVar.set('X')
            elif self.focus_get() == self.entryS:
                self.selectedVar.set('S')
            elif self.focus_get() == self.entryG:
                self.selectedVar.set('G')
        except:
            pass
    
    def onEnter(self, event):
        self.showGraph()
        
    def histogram(self):
        # Show properties frame for histogram (and hide others)
        self.frameHistogram.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.frameTrend.grid_forget()
        self.frameScatter.grid_forget()
        self.frameBargraph.grid_forget()
        self.frameBoxplot.grid_forget()
        # Clear textInfo 
        self.textInfo.delete('1.0', tk.END)
        # Clear graph
        self.ax.clear()         
        # Create graph
        stats, tests = graph.graph_hist(self.entryY.get(), bins=self.entryBins.get(), select=self.entryS.get(), groupby=self.entryG.get(),
             cumulative=self.cumulative.get(), percentage=self.percentage.get(), histtype=self.histtype.get(),
             xlim=None, ylim=None, size=(10, 6), ax=self.ax)
        # Show stats
        self.textInfo.insert('1.0', stats) 
        # Show tests
        self.textInfo.insert(tk.END, '\n\n')
        if not tests.empty:
            self.textInfo.insert(tk.END, tests)
        # Update canvas
        self.canvas.show()

    def trend(self):
        # Show properties frame for trend (and hide others)
        self.frameTrend.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.frameHistogram.grid_forget()
        self.frameScatter.grid_forget()
        self.frameBargraph.grid_forget()
        self.frameBoxplot.grid_forget()
        # Clear textInfo 
        self.textInfo.delete('1.0', tk.END)
        # Clear graph
        self.ax.clear()         
        # Create trend graph
        stats = graph.graph_trend(self.entryY.get(), self.entryX.get(), select=self.entryS.get(), groupby=self.entryG.get(), 
                                     mf=self.movingFunction.get(), mf_type=self.movingFunctionType.get(), mf_window=int(self.entryWindowSize.get()), mf_with_data=self.showData.get(), 
                                     constant=None, xlim=None, ylim=None, size=(10, 6), ax=self.ax)
        # Show stats
        self.textInfo.insert('1.0', stats)
        # Update canvas
        self.canvas.show()  

    def scatter(self):
        # Show properties frame for scatter graph (and hide others)
        self.frameScatter.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.frameHistogram.grid_forget()
        self.frameTrend.grid_forget()
        self.frameBargraph.grid_forget()
        self.frameBoxplot.grid_forget()
        # Clear textInfo 
        self.textInfo.delete('1.0', tk.END)
        # Clear old graph
        self.ax.clear()
        # Create scatter graph
        stats, models, info = graph.graph_xy(self.entryY.get(), self.entryX.get(), select=self.entryS.get(), groupby=self.entryG.get(), 
                                                bissectrice=self.bissectrice.get(), regression=self.regression.get(), constant=None, xlim=None, ylim=None, size=(10, 6), ax=self.ax)
        # Show stats and models info
        self.textInfo.insert('1.0', info)
        # Update canvas
        self.canvas.show()
        return
        
    def bargraph(self):
        # Show properties frame for bar graph (and hide others)
        self.frameBargraph.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.frameHistogram.grid_forget()
        self.frameTrend.grid_forget()
        self.frameScatter.grid_forget()
        self.frameBoxplot.grid_forget()
        # Clear textInfo 
        self.textInfo.delete('1.0', tk.END)
        # Clear old graph
        self.ax.clear() 
            
        if len(self.values(self.entryX).unique()) < 100:
            # Create graph
            stats = graph.graph_hist(self.entryY.get(), bins=self.entryBins.get(), select=self.entryS.get(), groupby=self.entryG.get(),
                                        cumulative=self.cumulative.get(), percentage=self.percentage.get(), histtype=self.histtype.get(),
                                        xlim=None, ylim=None, size=(10, 6), ax=self.ax)
            # Show stats
            self.textInfo.insert('1.0', stats)    
        else:
            messagebox.showinfo('Warning','Too many categories')
        # Update canvas
        self.canvas.show()
        return
    
    def boxplot(self):
        # Show properties frame for boxplot (and hide others)
        self.frameBoxplot.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.frameHistogram.grid_forget()
        self.frameTrend.grid_forget()
        self.frameScatter.grid_forget()
        self.frameBargraph.grid_forget()
        # Clear textInfo 
        self.textInfo.delete('1.0', tk.END)
        # Clear old plot
        self.ax.clear() 
        # Create graph
        if self.violinplot.get():
            stats = graph.graph_violinplot(self.entryY.get(), select=self.entryS.get(), groupby=self.entryG.get(),
                                           xlim=None, ylim=None, size=(10, 6), ax=self.ax)
        else:
            # normal boxplot - if  no bins
            if( self.nBinsBoxplot.get() == '' or ',' in self.entryY.get() ): 
                stats = graph.graph_boxplot(self.entryY.get(), select=self.entryS.get(), groupby=self.entryG.get(),
                                               xlim=None, ylim=None, size=(10, 6), ax=self.ax, srt=self.sortBoxplot.get() )
            # mono-boxplot - when bins and 1 variable
            elif( (',' not in self.entryY.get()) and len(self.entryY.get()) > 0 ): 
                stats = graph.graph_mono_boxplot(self.entryY.get(), select=self.entryS.get(), bins=self.nBinsBoxplot.get(),
                                                   xlim=None, ylim=None, title='', size=(10, 6), ax=self.ax)
        
        # Show stats
        self.textInfo.insert('1.0', stats)    
        # Update canvas
        self.canvas.show()
        return

    def showGraph(self):
        if self.graphtype.get() == 'H': 
            self.histogram()
        elif self.graphtype.get() == 'T': 
            self.trend()
        elif self.graphtype.get() == 'S': 
            self.scatter()
        elif self.graphtype.get() == 'R': 
            self.bargraph()
        elif self.graphtype.get() == 'B': 
            self.boxplot()

    def values(self, obj):
        if len(self.entryS.get()) > 0:
            # values, filtered
            data = eval('(' + obj.get() + ')[' + self.entryS.get() +']')
        else:
            # values, without filter
            data = eval('(' + obj.get() + ')')
        # Show error message if no values found
        if data.count() == 0:
            messagebox.showinfo('Error','No data found')
        # return data
        return data

    def listVars(self):
        if settings.dataframes:
            self.comboboxDataframes['values'] = list(settings.dataframes.keys())
        else:
            #settings.dataframes = {k: v for k, v in globals().items() if isinstance(eval(k), pd.core.frame.DataFrame)}
            # Get dataframes from global variables and add it to settings.dataframes dictionary
            for key, value in globals().items():
                if isinstance(eval(key), pd.core.frame.DataFrame):
                    # Correct name of columns - replace spaces, etc
                    value.rename(columns=lambda x: "".join(c for c in x if c not in ('()<>[]{},./?;:''"\|-=+*&^%$#@!`~')).strip().replace(' ', '_'), inplace=True)
                    settings.dataframes[key] = value
            self.comboboxDataframes['values'] = list(settings.dataframes.keys())
        
    def ListColumns(self, event):
        # Clear list
        self.listboxColumns.delete(0, tk.END)
        # Get selected dataframe name
        df = self.comboboxDataframes.get()
        if df:
            var_names = settings.dataframes[df].columns.values
            # Like in SQL % will substitute group of chars and _ will substitute single char
            search_term = self.searchValue.get().replace('%', '.*').replace('_', '.')
            if search_term.strip():
                # Filtering listbox using search term from searchValue box 
                pattern = re.compile('^' + search_term + '.*$', re.IGNORECASE)
                list_of_matching_vars = [s for s in var_names if pattern.match(s)]
                for item in list_of_matching_vars:
                    self.listboxColumns.insert(tk.END, item)
            else:
                for item in var_names:
                    self.listboxColumns.insert(tk.END, item)

    # Callback function - Add selected variable
    def SelectColumn(self, event):
        # If selection is not empty
        if self.listboxColumns.curselection():
            if self.selectedVar.get() == 'Y': 
                self.addVar(self.entryY)
            elif self.selectedVar.get() == 'X': 
                self.addVar(self.entryX)
            elif self.selectedVar.get() == 'S': 
                self.addVar(self.entryS, selection=True)
            elif self.selectedVar.get() == 'G': 
                self.addVar(self.entryG)
            self.showGraph()
            
    # Add selected variable to selected text box
    def addVar(self, entry, selection=False):
        # Get selected indices
        items = self.listboxColumns.curselection()
        # Count the selected items
        n = len(items)
        # Start with the first
        i = 1
        # Add each variable to the entry
        for item in items:            
            #varname = self.comboboxDataframes.get() + '.' + self.listboxColumns.get(self.listboxColumns.curselection())
            varname = self.comboboxDataframes.get() + '.' + self.listboxColumns.get(item)
            if selection:
                text = "({}>0)".format(varname)
            else:
                text = varname
            if len(entry.get()) == 0:
                # If nothing yet there, simply add the variable
                entry.insert(0, text)
            elif entry.get()[-1] in ',+-*/':
                # If there is something already and it ends with , or + or - or * or /, then add the variable
                entry.insert(tk.END, ' ' + text)
            elif entry.get()[-1] in ' ':
                # If there is something already and there is an extra space, then add the variable
                entry.insert(tk.END, text)
            else:      
                # replace the existing text
                entry.delete(0, tk.END)
                entry.insert(0, text)
            # If it's not the last of the selection, add a comma
            if i < n:
                if selection:
                    entry.insert(tk.END, ' & ')
                else:
                    entry.insert(tk.END, ', ')
            # Increase the counter
            i += 1
            
    # Callback function - Show Python code
    def code(self):
        if self.buttonCode.cget('text') == 'Show Code':
            self.panedwindowMain.add(self.frameRight)
            self.buttonCode.configure(text = 'Hide Code')
        else:
            self.panedwindowMain.forget(self.frameRight)
            self.buttonCode.configure(text = 'Show Code')
			
    def OpenCommand(self):
        filename = tk.filedialog.askopenfilename(title='Supported file formats are: xls, xlsx, csv, pkl')
        if(filename == ""): # case of pressing ESC
            return  
        pos1 = filename.rfind('/')
        pos2 = filename.rfind('.')
        pos3 = filename.find(' ', pos1)
        if( pos3 == -1 ):
            dfName = CorrectString( filename[pos1+1:pos2] )
        else:
            dfName = CorrectString( filename[pos1+1:pos3] )
        extension = filename[pos2+1:]
        print('Opening file:', filename)  
        
        if (extension == 'xls') | (extension == 'xlsx'):
            xl = pd.ExcelFile(filename)
            nSheets = len(xl.sheet_names)
            if( nSheets == 1 ):
                exec("settings.dataframes['" + dfName +"'] = pd.read_excel('" + filename +"')")
            elif( nSheets > 1 ):
                sheetName = tkSD.askOptions( self, xl.sheet_names, 'Choose Excel sheet to import')
                if(sheetName == ""): # case of pressing ESC
                    return
                sheetName = CorrectString(sheetName)
                exec("settings.dataframes['" + sheetName +"'] = pd.read_excel('" + filename +"', sheet_name=sheetName)")
            self.comboboxDataframes.set(sheetName)
            
        elif(extension == 'pkl'):
            exec("settings.dataframes['" + dfName +"'] = pd.read_pickle('" + filename +"')")
            self.comboboxDataframes.set(dfName)
            
        elif(extension == 'csv'):
            exec("settings.dataframes['" + dfName +"'] = pd.read_csv('" + filename +"',sep=';')")
            self.comboboxDataframes.set(dfName)
            
        else:
            messagebox.showinfo("Unsupported file extension","Unsupported file extension: " + extension)
        
        # Correct name of columns - replace spaces, etc 
        for dataframe in settings.dataframes:
            cols = settings.dataframes[dataframe].columns.values.tolist()
            for name in cols:
                new = CorrectString( name )
                if(new != name):
                    settings.dataframes[dataframe].rename(columns={name: new}, inplace=True)
        self.comboboxDataframes.set(self.comboboxDataframes.get())
        self.ListColumns(self.comboboxDataframes) # impoirtant to refresh
            
    def RenameDatasetCommand(self):
        oldName = self.comboboxDataframes.get()
        newName = tkSD.askstring("New name for dataset", "enter")
        settings.dataframes[newName] = settings.dataframes.pop(oldName)
        self.comboboxDataframes.set(newName)
        
    def RenameVariableCommand(self):
        item = list(self.listboxColumns.curselection())[0]
        oldName = self.listboxColumns.get(item)
        newName = tkSD.askstring("New name for variable", "enter")
        settings.dataframes[self.comboboxDataframes.get()] = settings.dataframes[self.comboboxDataframes.get()].rename( index=str, columns={oldName:newName} )
        self.ListColumns(self.comboboxDataframes)   
        
    def PreferencesCommand(self):
        messagebox.showinfo("TO BE DONE",
                            "Required preferences need to be discussed\n"
                            "Before Implementation")
        
    def AboutPydexCommand(self):
        messagebox.showinfo("About pydex",
                            "Python Data Exploration Tool (PyDex)\n"
                             "by ArcelorMittal Poland\n"
                             "Main Contributors:\n"
                             "Luc Van De Putte (luc.vandeputte@arcelormittal.com)\n"
                             "Tomasz Czaja (Tomasz.Czaja2@arcelormittal.com)\n"
                             "Grzegorz Kępisty (grzegorz.kepisty@arcelormittal.com)" )
                   
    # Callback function - Handle exceptions
    def HandleExceptionCommand(self, *args):
        err = tb.format_exception(*args)
        messagebox.showinfo('Exception',err)
				   
    def createWidgets(self):  
        # Create top classic menubars
        menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=menu)
        menu.add_command(label="Open ...", command=self.OpenCommand)
        menu.add_command(label="Exit", command=self.master.destroy)
        menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=menu)
        menu.add_command(label="Rename dataset", command=self.RenameDatasetCommand)
        menu.add_command(label="Rename variable", command=self.RenameVariableCommand)
        menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=menu)
        menu.add_command(label="Preferences", command=self.PreferencesCommand)
        menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=menu)
        menu.add_command(label="About", command=self.AboutPydexCommand)
        self.master.config(menu=self.menubar)
		
		 # Get top window 
        self.top = self.winfo_toplevel()
        
        # Make it stretchable         
        self.top.rowconfigure(0, weight=1)
        self.top.columnconfigure(0, weight=1)

        # Make certain rows and columns stretchable
        self.rowconfigure(0, weight=1)        
        self.columnconfigure(0, weight=1)

        #%% Main paned window
        # Paned window (stack horizontal, i.e. next to each other)
        self.panedwindowMain = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.panedwindowMain.grid(row = 0, column = 0, sticky = tk.NE + tk.SW, padx =1, pady=1)
        
        # Create frameLeft and add it to the paned window
        self.frameLeft = ttk.Frame(self.panedwindowMain)
        self.panedwindowMain.add(self.frameLeft)
        self.panedwindowMain.paneconfigure(self.frameLeft, sticky=tk.NW+tk.SE)
        # Make frameLeft stretchable
        self.frameLeft.rowconfigure(1, weight=1)
        self.frameLeft.columnconfigure(1, weight=1)

        # Create frameMiddle and add it to the paned window
        self.frameMiddle = ttk.Frame(self.panedwindowMain)
        self.panedwindowMain.add(self.frameMiddle)
        self.panedwindowMain.paneconfigure(self.frameMiddle, sticky=tk.NW+tk.SE)
        # Make frameMiddle stretchable
        #self.frameMiddle.rowconfigure(0, weight=1)
        self.frameMiddle.rowconfigure(1, weight=1)
        self.frameMiddle.columnconfigure(8, weight=1)
        
        # Create frameRight and add it to the paned window
        self.frameRight = ttk.Frame(self.panedwindowMain)
        self.panedwindowMain.add(self.frameRight)
        self.panedwindowMain.paneconfigure(self.frameRight, sticky=tk.NW+tk.SE)
        self.panedwindowMain.forget(self.frameRight)
        # Make frameRight stretchable
        self.frameRight.rowconfigure(1, weight=1)
        self.frameRight.columnconfigure(0, weight=1)

        #%% LEFT
        # Combobox to select a dataframe
        self.comboboxDataframes = ttk.Combobox(self.frameLeft, postcommand=self.listVars)
        self.comboboxDataframes.grid(row = 0, column = 0, columnspan=3, sticky = tk.NE + tk.SW, padx =5, pady=5)
        self.comboboxDataframes.bind('<Return>', self.ListColumns)
        self.comboboxDataframes.bind('<<ComboboxSelected>>',self.ListColumns)
        
        # listbox with scrollbar to select a variable (column in the data frame)
        self.frameListbox = ttk.Frame(self.frameLeft, relief='solid', borderwidth=1)
        self.frameListbox.grid(row = 1, column = 0, columnspan=2, sticky = tk.NE + tk.SW, padx =5, pady=5)
        self.frameListbox.rowconfigure(0, weight=1)
        self.frameListbox.columnconfigure(0, weight=1)
        # Listview of columns in the dataframe
        self.scrollbarColumns = ttk.Scrollbar(self.frameListbox, orient=tk.VERTICAL)
        self.listboxColumns = tk.Listbox(self.frameListbox, selectmode = tk.EXTENDED, yscrollcommand=self.scrollbarColumns.set)
        self.listboxColumns.bind("<B1-Leave>", lambda event: "break") # removes listbox bug with horizontal text shifting 
        self.scrollbarColumns.config(command=self.listboxColumns.yview)
        self.scrollbarColumns.grid(row=0, column=1, sticky =tk.NW+tk.SE)
        self.listboxColumns.bind('<<ListboxSelect>>',self.SelectColumn)
        self.listboxColumns.grid(row=0, column = 0, sticky = tk.NE + tk.SW)

        # Search field
        ttk.Label(self.frameLeft, text = "Find").grid(row = 14, column = 0, sticky = tk.W, padx =5, pady=5)
        self.searchValue = tk.StringVar()
        self.searchValue.trace("w", lambda name, index, mode, event=None: self.ListColumns(event))
        #self.entryFind = ttk.Entry(self, textvariable=self.searchValue, command=self.ListColumns(None))
        self.entryFind = ttk.Entry(self.frameLeft, textvariable=self.searchValue)
        self.entryFind.grid(row = 14, column = 1, columnspan=2, sticky=tk.W+tk.E, padx =5, pady=5)

        # Properties frame
        self.frameProperties = ttk.Frame(self.frameLeft, relief='ridge', borderwidth=4)
        self.frameProperties.grid(row = 15, column = 0, columnspan=3, sticky = tk.W+tk.E, padx=5, pady=5)
        
        # Properties frame Histogram
        self.frameHistogram = ttk.Frame(self.frameProperties)
        self.frameHistogram.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.labelHistogram = ttk.Label(self.frameHistogram, text='Number of bins:')
        self.labelHistogram.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.entryBins = ttk.Entry(self.frameHistogram, justify='center')
        self.entryBins.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.entryBins.bind('<Return>', self.onEnter)
        
        self.histtype = tk.StringVar() 
        self.histtype.set('bar') #from matplotlib doc: histtype : {‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’},
        self.optionHist   = ttk.Radiobutton(self.frameHistogram, text='Histogram ', variable=self.histtype, value='bar', command=self.histogram)
        self.optionStairs = ttk.Radiobutton(self.frameHistogram, text='Stairs    ', variable=self.histtype, value='step', command=self.histogram)
        self.optionHist.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.optionStairs.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.cumulative = tk.IntVar()
        self.checkCumulative = ttk.Checkbutton(self.frameHistogram, text='Cumulative', variable=self.cumulative, command=self.histogram)
        self.checkCumulative.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
            
        self.percentage = tk.IntVar()
        self.checkPercentage = ttk.Checkbutton(self.frameHistogram, text='Percentage', variable=self.percentage, command=self.histogram)
        self.checkPercentage.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
            
        self.histtest = tk.IntVar()
        self.checkHisttest = ttk.Checkbutton(self.frameHistogram, text='t-test + F-test/Z-test', variable=self.histtest, command=self.histogram)
        self.checkHisttest.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Properties frame Trend 
        self.frameTrend = ttk.Frame(self.frameProperties)
        self.frameTrend.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.frameWindow = ttk.Frame(self.frameTrend)
        self.frameWindow.grid(row=1, column=0, sticky=tk.W, padx=0, pady=0)
        
        self.movingFunction = tk.BooleanVar()
        self.movingFunctionType = tk.StringVar()
        self.movingFunctionType.set('mean')

        self.checkMoving = ttk.Checkbutton(self.frameTrend, text='Moving function', variable=self.movingFunction, command=self.trend)
        self.checkMoving.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.labelWindow = ttk.Label(self.frameWindow, text='Window size:')
        self.labelWindow.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        # Entry to define size of data window for calculationg moving function
        self.entryWindowSize = ttk.Entry(self.frameWindow, width=10)
        self.entryWindowSize.insert(0, '50')
        self.entryWindowSize.grid(row=0, column=1, sticky=tk.W, padx=0, pady=5)
        self.entryWindowSize.bind('<Return>', self.onEnter)
    
        # Moving function selection
        self.optionMovingAverage = tk.Radiobutton(self.frameTrend, text='Average', variable=self.movingFunctionType, value='mean', command=self.trend)
        self.optionMovingAverage.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.optionMovingMedian = tk.Radiobutton(self.frameTrend, text='Median', variable=self.movingFunctionType, value='median', command=self.trend)
        self.optionMovingMedian.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.optionMovingStdDev = tk.Radiobutton(self.frameTrend, text='StdDev', variable=self.movingFunctionType, value='std', command=self.trend)
        self.optionMovingStdDev.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.optionMovingSum = tk.Radiobutton(self.frameTrend, text='Sum', variable=self.movingFunctionType, value='sum', command=self.trend)
        self.optionMovingSum.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        # Checkbox indicating if original data should be also plotted
        self.showData = tk.BooleanVar()
        self.showData.set(True)
        self.checkMovingWithData = ttk.Checkbutton(self.frameTrend, text='Show original values', variable=self.showData, command=self.trend)
        self.checkMovingWithData.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.frameTrend.grid_forget()
        
        # Properties frame Scatter
        self.frameScatter = ttk.Frame(self.frameProperties)
        self.frameScatter.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.bissectrice = tk.IntVar()
        self.checkBissectrice = ttk.Checkbutton(self.frameScatter, text='Bissectrice', variable=self.bissectrice, command=self.scatter)
        self.checkBissectrice.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.regression = tk.IntVar()
        self.checkRegression = ttk.Checkbutton(self.frameScatter, text='Regression', variable=self.regression, command=self.scatter)
        self.checkRegression.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.frameScatter.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        # Properties panel Bargraph
        self.frameBargraph = ttk.Frame(self.frameProperties)
        self.frameBargraph.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.labelAggregateFunction = ttk.Label(self.frameBargraph, text='Function')
        self.labelAggregateFunction.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.comboboxAggregateFunction = ttk.Combobox(self.frameBargraph)
        self.listFunctions = ['count','mean','sum','min','max','stdev']
        self.comboboxAggregateFunction['values'] = self.listFunctions
        self.comboboxAggregateFunction.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
                
        self.frameBargraph.grid_forget()
        
        # Properties panel Boxplot
        self.frameBoxplot = ttk.Frame(self.frameProperties)
        self.frameBoxplot.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.violinplot = tk.IntVar()
        self.checkViolinplot = ttk.Checkbutton(self.frameBoxplot, text='Violin plot', variable=self.violinplot, command=self.boxplot)
        self.checkViolinplot.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.sortBoxplot = tk.IntVar()
        self.checkSortBx = ttk.Checkbutton(self.frameBoxplot, text='Sort boxplot', variable=self.sortBoxplot, command=self.boxplot)
        self.checkSortBx.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.nBinsBoxplot = tk.StringVar(value='')
        self.selectNBins   = ttk.Entry(self.frameBoxplot, text='Bins', textvariable=self.nBinsBoxplot  )
        self.selectNBins.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.frameBoxplot.grid_forget()
        #%% MIDDLE
        # Chart type
        self.graphtype = tk.StringVar() 
        self.graphtype.set('H')
        ttk.Radiobutton(self.frameMiddle, text='Histogram ', variable=self.graphtype, value='H', command=self.histogram).grid(row = 0, column = 0, sticky = tk.W, padx =5, pady=5)
        ttk.Radiobutton(self.frameMiddle, text='Trend     ', variable=self.graphtype, value='T', command=self.trend).grid(row = 0, column = 1, sticky = tk.W, padx =5, pady=5)
        ttk.Radiobutton(self.frameMiddle, text='XY        ', variable=self.graphtype, value='S', command=self.scatter).grid(row = 0, column = 2, sticky = tk.W, padx =5, pady=5)
        ttk.Radiobutton(self.frameMiddle, text='Bar       ', variable=self.graphtype, value='R', command=self.bargraph).grid(row = 0, column = 3, sticky = tk.W, padx =5, pady=5)
        ttk.Radiobutton(self.frameMiddle, text='Boxplot   ', variable=self.graphtype, value='B', command=self.boxplot).grid(row = 0, column = 4, sticky = tk.W, padx =5, pady=5)

        self.buttonRefresh = ttk.Button(self.frameMiddle, text='Refresh', command=self.showGraph)
        self.buttonRefresh.grid(row = 0, column = 9, sticky = tk.W, padx =5, pady=5)
        self.buttonCode = ttk.Button(self.frameMiddle, text='Show Code', command=self.code)
        self.buttonCode.grid(row = 0, column = 10, sticky = tk.W, padx =5, pady=5)

        # Middle paned window (stack vertical, i.e. under each other)
        self.panedwindowMiddle = tk.PanedWindow(self.frameMiddle, orient=tk.VERTICAL)
        self.panedwindowMiddle.grid(row = 1, column = 0, columnspan=11, sticky = tk.NE + tk.SW, padx =1, pady=1)
        
        # Create matplotlib figure and add it to frameMiddle
        self.fig = Figure(figsize=(10, 7), tight_layout=True)
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.ax1.set_title('title 1')
        self.ax1.set_xlabel( 'X-axis' )
        self.ax1.set_ylabel( 'Y-axis' )
        #self.ax1.grid(True)
        
#        self.ax2 = self.fig.add_subplot(2,2,2)
#        self.ax2.set_title('title 2')
#        self.ax2.set_xlabel( 'X-axis' )
#        self.ax2.set_ylabel( 'Y-axis' )
#        self.ax2.grid(True)
#            
#        self.ax3 = self.fig.add_subplot(2,2,3)
#        self.ax3.set_title('title 3')
#        self.ax3.set_xlabel( 'X-axis' )
#        self.ax3.set_ylabel( 'Y-axis' )
#        self.ax3.grid(True)
            
        self.ax = self.ax1
        self.ax.set_facecolor('#FFFFE0')
                               
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameMiddle)
        
		# creates interactive toolbar
        self.toolbar_frame = tk.Frame(self.frameMiddle) # required to bypass grid/pack conflict
        self.toolbar_frame.grid(row = 0, column = 5, sticky = tk.W, padx =5, pady=5)
        self.toolbar = CustomToolbar(self.canvas, self.toolbar_frame, self.showGraph) 

        self.panedwindowMiddle.paneconfigure(self.canvas.get_tk_widget(), sticky=tk.NW+tk.SE)
        self.panedwindowMiddle.add(self.canvas.get_tk_widget())   

        #self.cid = self.canvas.mpl_connect('button_press_event', self)

        # Create frameSelection and add it to the paned window
        self.frameSelection = ttk.Frame(self.panedwindowMiddle, relief='ridge', borderwidth=4)
        # Make frameSelection stretchable
        self.frameSelection.rowconfigure(4, weight=1)
        self.frameSelection.columnconfigure(1, weight=1)
                
        # Y, X, Select and Group by labels
        self.selectedVar = tk.StringVar()
        self.selectedVar.set('Y')
        self.radiobuttonY = ttk.Radiobutton(self.frameSelection, text='Y       ', variable=self.selectedVar, value='Y')
        self.radiobuttonY.grid(row=0, column=0, sticky = tk.NE + tk.SW, padx =1, pady=1)
        self.radiobuttonX = ttk.Radiobutton(self.frameSelection, text='X       ', variable=self.selectedVar, value='X')
        self.radiobuttonX.grid(row=1, column=0, sticky = tk.NE + tk.SW, padx =1, pady=1)
        self.radiobuttonS = ttk.Radiobutton(self.frameSelection, text='Select  ', variable=self.selectedVar, value='S')
        self.radiobuttonS.grid(row=2, column=0, sticky = tk.NE + tk.SW, padx =1, pady=1)
        self.radiobuttonG = ttk.Radiobutton(self.frameSelection, text='Group by', variable=self.selectedVar, value='G')
        self.radiobuttonG.grid(row=3, column=0, sticky = tk.NE + tk.SW, padx =1, pady=1)
        
        # Y, X, Select and Group by entries
        self.entryY = ttk.Entry(self.frameSelection)
        self.entryY.grid(row=0, column=1, sticky = tk.NE + tk.SW, padx =1, pady=1)
        self.entryY.bind('<Return>', self.onEnter)
        self.entryX = ttk.Entry(self.frameSelection)
        self.entryX.grid(row=1, column=1, sticky = tk.NE + tk.SW, padx =1, pady=1)
        self.entryX.bind('<Return>', self.onEnter)
        self.entryS = ttk.Entry(self.frameSelection)
        self.entryS.grid(row=2, column=1, sticky = tk.NE + tk.SW, padx =1, pady=1)
        self.entryS.bind('<Return>', self.onEnter)
        self.entryG = ttk.Entry(self.frameSelection)
        self.entryG.grid(row=3, column=1, sticky = tk.NE + tk.SW, padx =1, pady=1)
        self.entryG.bind('<Return>', self.onEnter)
        
        # Text area for Info
        self.textInfo = ScrolledText(self.frameSelection, height=4) 
        self.textInfo.grid(row=4, column=0, columnspan=2, sticky = tk.NE + tk.SW, padx =1, pady=1)

        self.panedwindowMiddle.add(self.frameSelection, minsize=80)
        self.panedwindowMiddle.paneconfigure(self.frameSelection, sticky=tk.NW+tk.SE)

#        # Text area for Info
#        self.textInfo = ScrolledText(self.panedwindowMiddle, height=4) 
#        self.panedwindowMiddle.add(self.textInfo)
#        self.panedwindowMiddle.paneconfigure(self.textInfo, sticky=tk.NW+tk.SE)
       
        #%% RIGHT
        # Code
        self.codePath = tk.StringVar()
        self.codePath.set(r"D:\pydex_" + str(datetime.datetime.today())[0:16].replace('-','_').replace(' ','_').replace(':','') + ".py")
        ttk.Label(self.frameRight, text = "Code", textvariable=self.codePath).grid(row = 0, column = 0, sticky = tk.NW+tk.SE, padx =5, pady=5)

        # Text area for generated code
        self.textCode = ScrolledText(self.frameRight, height=4) 
        self.textCode.grid(row = 1, column = 0, sticky = tk.NW+tk.SE, padx =5, pady=5)
        self.textCode.insert(tk.END, '""" Data exploration')
        self.textCode.insert(tk.END, '\n\n')
        self.textCode.insert(tk.END, '@date: ' + str(datetime.date.today()))
        self.textCode.insert(tk.END, '\n\n')
        self.textCode.insert(tk.END, '@author: ' + getpass.getuser())
        self.textCode.insert(tk.END, '\n"""')
        self.textCode.insert(tk.END, '\n')
        self.textCode.insert(tk.END, '\n# Import libraries')
        self.textCode.insert(tk.END, '\nimport arcelormittal as am')
        self.textCode.insert(tk.END, '\n')
        self.textCode.insert(tk.END, '\n# Get data')
        self.textCode.insert(tk.END, "\nd = am.data('XYZ')")

class CustomToolbar(NavigationToolbar2TkAgg):
    def __init__(self,canvas_,parent_, commHome):
        self.commHome = commHome
        self.toolitems = (
            ('Home', "Reset original view\n\ ('h' or 'r')", 'home', 'home'),
            ('Back', "previous view\n\ ('c' or 'left arrow' or 'backspace')", 'back', 'back'),
            ('Forward', "next view\n\ ('v' or 'left arrow')", 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', "Pan axis (left mouse), Zoom (right mouse).\n\ ('p')", 'move', 'pan'),
            ('Zoom', "Zoom box\n\ ('o')", 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            ('Subplots', 'Configure subplots axis', 'subplots', 'configure_subplots'),
            ('Save', 'Save plot', 'filesave', 'save_figure'),
            )
        NavigationToolbar2TkAgg.__init__(self,canvas_,parent_)
    def set_message(self, msg): # this is needed to avoid printing coordinates
        pass
    def home(self, *args): # override default function to work correctly
        self.commHome() 

def CorrectString( sss ):
    new = str(sss)
    new = ''.join(new.strip().split()).replace(" ", "_")
    new = ''.join([c if c not in ('()<>[]{},./?;:''"\|-=+*&^%$#@!`~') else '_' for c in new ])
    return new

# Allow the class to run stand-alone.
if __name__ == "__main__":
    app = explore() 
    app.master.title('PyExplore - Python Data Exploration Tool - version 0.1')
    app.mainloop()