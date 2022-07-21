# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


age = 15
rate = .01
agestarts = 25

def piecewise(age, rate, agestarts):
    out = 0 if age < agestarts else rate * (age - agestarts+1) 
    out = 1 if out >1 else out
    return(out)
        
piecewise(age,rate,agestarts)
out = piecewise(age,rate,agestarts)