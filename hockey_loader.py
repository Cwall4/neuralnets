# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:55:38 2017

@author: Colin
"""

import numpy as np
import csv
import pandas as pd

def load_data():
    #f = csv.reader('lhjmq_game_49.csv')
    #f = np.loadtxt('nhl_player_mother.csv', delimiter=',', skiprows=1, usecols=(3,))
    f = pd.read_csv('nhl_player_mother.csv')
    #print(f['to'])
    f['y'] = f['to'] - f['birthyear']
    f = f.sort_values(by = ['player', 'age'])
    f['yrsPlayed'] = f['year'] - f['from'] + 1
    fp = f.groupby(['player'], as_index = False).cumsum()
    f['gp_cum'] = fp['gp']
    f['g_cum'] = fp['g']
    f['a_cum'] = fp['a']
    print(fp.shape)
    #f = pd.merge(f, fp[['player', 'ppg']], on = ['player'], how = 'left', suffixes = ['_a', '_m'])
    f = f.drop(f.columns[[2, 4, 5, 19, 20, 24, 26]], 1)
    #print(f)
    f = f.values
    return(f)

def load_goalie_data():
    f = pd.read_excel('Goalie data.xlsx')
    #print(f['to'])
    f['Year'] = f['Season'].str[0:4].astype('float')
    #f['y'] = f['to'] - f['birthyear']
    fp = f.groupby(['Player'], as_index = False)
    fpmax = fp.max()
    fpmax['RetAge'] = fpmax['Age']
    fm = pd.merge(f, fpmax[['Player', 'RetAge']])
    #f = f.sort_values(by = ['player', 'age'])
    #f['yrsPlayed'] = f['year'] - f['from'] + 1
    #fp = f.groupby(['player'], as_index = False).cumsum()
    #f['gp_cum'] = fp['gp']
    #f['g_cum'] = fp['g']
    #f['a_cum'] = fp['a']
    #print(fp.shape)
    #f = pd.merge(f, fp[['player', 'ppg']], on = ['player'], how = 'left', suffixes = ['_a', '_m'])
    fm = fm.drop(fm.columns[[0, 1, 2, 3, 4, 5, 27]], 1)
    #print(f)
    #f = f.values
    return(fm)
