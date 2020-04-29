'''
Description: Functions to calculate and plot polinomial regressions
for coronavirus cases

author: daniel.vasconcelosgomes at gmail com
29/04/2020

Usage:

python na.py CSV_FILE Title Days_since_start
eg:

> python na.py sp.csv SP 57


[Polinomial regression level: 8 ]
RMSE: 282.76527302319323
R^2:  0.9983290353285625
Maximum cases ( SP ) estimated: 143816.35359523178 at 79 days.
Time to double: [7.29528157] days


References and sources:

https://www.python-course.eu/numerical_programming_with_python.php
https://www.analyticsvidhya.com/blog/2020/03/polynomial-regression-python/
https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
https://www.math.ust.hk/~machas/mathematical-biology.pdf
https://www.researchgate.net/publication/305032108_The_mathematical_modelling_of_population_change
https://covid.saude.gov.br/
https://www.seade.gov.br/coronavirus/
http://coronavirus.dadosecausas.org/
https://ourworldindata.org/coronavirus
https://docs.google.com/spreadsheets/d/1gR1QaUdWCclJxE4q1huY-SGc4NxbWjeVUscSmvMO8d4/edit?usp=sharing



'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import os
import sys


'''
Reads a csv file and looks for the columns t and cases
returning two panda data sets
'''
def read_data( filename ):

    dataset = pd.read_csv(filename)
    t = dataset[['t']]
    y = dataset[['cases']]
    
    return t,y
    

'''
Reads a csv file and looks for the columns t and h1/h2
returning 3 panda data sets
'''   
def read_data_ex( filename , h1, h2 ):

    dataset = pd.read_csv(filename)
    t = dataset[['t']]
    y1 = dataset[[h1]]
    y2 = dataset[[h2]]
    
    return t,y1,y2
       
 
'''
Based on two data sets plots the graphic yxt
using title and labels for reference
''' 
def print_data( y, t, color,  title, x_label, y_label ):

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(  t, y, color  )
    plt.show()
 

'''
Based on 3 data sets plots the graphic y1 x t and y2 x t
using title and labels for reference
''' 
def print_data_ex(  y1, y2, t, title, x_label, y_label ):

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot( t, y1, 'g'  )
    plt.plot( t, y2, 'b'  )
    plt.show()


'''
Based on two data calculates the polinomial of grade degree based on dataset yxt
'''  
def pol_regression( y, X, _degree_ ):
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    poly_reg = PolynomialFeatures(degree=_degree_)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    return pol_reg,poly_reg
    

'''
Based predicted data and real data calculates the RMSE and R^2 values
printing on screen
'''
def show_error( y, y_poly_pred ):

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)
    print( "RMSE: {}".format(rmse))
    print( "R^2:  {}".format(r2))



'''
Based on two data sets plots the graphic yxt
using title and labels for reference
'''
def print_regression( X, y, pol_reg, poly_reg, the_title, x_l, y_l ):

    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title(the_title)
    plt.xlabel(x_l)
    plt.ylabel(y_l)
    plt.show() 


'''
Calculates the time required so the cases double
'''
def time_to_double( today, ys ):

    p = ys[ today + 1 ]  
    pmax = ys.max()
    tmax = ys.argmax() + 1 
    
    if 2*p > pmax:
        print("Cannot estimate")
    else:
        t =  (( 2*p - ys[ today -1 ] ) / ( p - ys[ today - 1] ))*1
        print("Time to double: {} days".format(t) )
        
        

'''
Polinomial regression main function
'''
def do_pol_reg( t, y, msg, level, today  ):

    print("[Polinomial regression level: {} ]".format( level ) )
    pol_reg,poly_reg = pol_regression(  y, t, level )   
    y_poly_pred = pol_reg.predict(poly_reg.fit_transform( t ))
    show_error( y, y_poly_pred )
    ayear = pd.DataFrame( [ i for i in range( 1,360 ) ] ) 
    ys = pol_reg.predict( poly_reg.fit_transform( ayear ))
    
    
    print("Maximum cases ( {} ) estimated: {} at {} days.".format( msg, ys.max(), ys.argmax() + 1) )
    time_to_double( today, ys )
    
    #print_regression( t, y, pol_reg, poly_reg, "Cases Corona", "days", "Cases" )    
    

'''
Based on two data sets yxt calculates the best polinomial that gives
the shortest RMSE
'''
def find_min_error( t, y ):
    
    best_fit = []
    for level in range(1,20):
        pol_reg,poly_reg = pol_regression(  y, t, level )   
        y_poly_pred = pol_reg.predict(poly_reg.fit_transform( t ))
        best_fit.append( np.sqrt(mean_squared_error(y,y_poly_pred)))
        
    return best_fit.index( min(best_fit) ) + 1

  
def main(argv):

    os.system('cls')

    if len(argv) <= 1:
        print("Wrong execution.")
        return      
       
    filename = argv[1]
    option   = argv[2]
    today    = int( argv[3] )
            
    if (  os.path.exists(filename) == False ):
    
        print("csv file does not exist..")
        return
    
    t, y = read_data( filename )
    n = find_min_error ( t, y )  
    do_pol_reg( t, y, option , n, today )

                         
# main program start  ---------------------------------------------    
                 
if __name__ == '__main__':
    main(sys.argv)  

# main program end  -----------------------------------------------       
