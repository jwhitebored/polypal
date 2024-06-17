# -*- coding: utf-8 -*-
"""
June 2024
@author: jwhitebored
github: https://github.com/jwhitebored/polypal.git

NOTE: I'm currently looking for work in data science, python development, or
      math-related fields. If you're hiring, shoot me an email at
      james.white2@mail.mcgill.ca
"""
import numpy as np
import pandas as pd

######################### SIGNAL GENERATION ###################################
path = 'C:/Users/'

#TLDR: The 60,000 polynomials (aka signals) are generated and...
#Have 1024 data points
#Exist on a random domain within (-30, 30)
#Have uniformly random integer degrees (0 through 9)
#Have uniformly random coefficients (-10 through 10)


def batch_maker(bn):
    batchnumber = bn
    maxdeg = 9 #The max degree of any polynomial that will be generated.
    maxcoeff = 10 #The max coefficient of any polynomial that will be generated.
    numpolys = 60000 #The number of polynomials that will be generated.
    train_num = 50000 #The proportion of polynomials that will be used for training data.
    test_num = numpolys-train_num
    
    #Generates a list of random integer degrees for each polynomial (0-9).
    #The degrees are uniformly distributed.
    #This is will be the y_train/y_test data.
    polydegs = np.array([int((maxdeg+1)*i) for i in np.random.rand(numpolys)])
    
    #Generates random coefficients for each polynomial according to their degree.
    #Coefficients range from -maxcoeff to +maxcoeff (-10,10).
    #Coefficients of 0 are made for degrees higher than the max degree of the polynomial.
    polycoeffs = (np.array([np.concatenate((
                 [2*maxcoeff*i-maxcoeff for i in np.random.rand(polydegs[j]+1)],
                 [0 for i in range(maxcoeff-polydegs[j])]), axis=0)
                 for j in range(numpolys)]))
    
    #Convert the coefficients to float32, reducing file size/computation load.
    polycoeffs = np.float32(polycoeffs)
    
    #Length (# of data points) of sample from polynomial.
    signallength = 1024
    
    #The positive (and negative) bound of the polynomial's domain
    #i.e. the domain lies somewhere on (-30,30), but where exactly is
    #randomized according a uniform distribution.
    #Note that this is a symmetric window of possible bounds, 
    #but the bounds themselves are random, so not necessarily symmetric across the y-axis
    domain_bound = 30
    
    #Generates the xvals on a random domain to be used for creating the polynomial
    xval_start_stop_list = np.float32(np.array([[domain_bound*(2*j-1) for j in np.sort(np.random.rand(2))] for i in range(numpolys)])) #Used to generate random signal domains in linspace at *here*
    
    #Function that takes an x value, and generates a y value according to the
    #degree, coefficients, and domain defined for the polynomial.
    #Note that the 0th coeff is for the x^0 power and so on.
    def polygenerator(deg, coeffs, xval):
        yval = 0
        for i in range(deg+1):
            yval = yval + coeffs[i]*np.power(xval, i)
        return yval
    
    #Generates the polynomial y_data, taking in the arrays for the degrees,
    #the coefficients, and the domains, and spitting out an NxM array, where
    #N=num_polys and M=signallength.
    def signalgenerator(polydegs, polycoeffs, signallength=1024):
        signals = np.array([np.array([polygenerator(polydegs[i], polycoeffs[i], j) 
                            for j in np.linspace(xval_start_stop_list[i][0], xval_start_stop_list[i][1], num=signallength)]) 
                            for i in range(numpolys)])
        return np.float32(signals)
    
    #A key feature of noise in a system is the signal to noise ration (SNR)
    #This generates a uniformly random SNR ranging from 0 to 1 for each polynomial,
    #i.e. from no noise to equal parts noise and signal
    snr = np.random.rand(numpolys)
    
    #Adds noise to a generated NxM array of signals according to the snr values
    #assigned for each signal. The SNR is proportional to each signal's strength
    #i.e. proportional to the y_value.
    def noiseadder(signals):
        noisysignals = np.array([np.array([j + np.random.normal(0, np.abs(snr[k]*j), None) for j in i]) for i,k in zip(signals, range(numpolys))])
        return np.float32(noisysignals)
    
    #Should the noise be proportional to the value of the polynomial?
    #It depends. This is the key concept here. The SNR is a random variable
    #unique to system. The dependency of the noise on the data can be tweaked here.
    #Refer to: https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    #Generates the noiseless signal
    freud = signalgenerator(polydegs, polycoeffs, signallength) #signal freud (bad pun, I know)
    
    #Generates the noisy signal
    noita = noiseadder(freud) #noisy signal array (and reference to an incredible game)
    

######################### Data Framing ########################################
    #Adds the noisy signals to a dataframe and saves to a csv file
    df = pd.DataFrame(np.array(noita))
    df.columns = ['T' + str(i) for i in range(signallength)]
    df.insert(0, 'P(x)', ['P_' + str(i) for i in range(numpolys)])
    df.insert(0, 'Degree', polydegs, True)
    df.to_csv(path + '50k_and_10k_signals_batch_' + str(batchnumber) + '.csv', index=False)
    
    #Adds the coefficients to a dataframe and saves to a csv file
    coefdf = pd.DataFrame(np.array(polycoeffs))
    coefdf.columns = ['x^' + str(j) for j in range(maxdeg+2)]
    coefdf.insert(0, 'P(x)', ['P_' + str(i) for i in range(numpolys)])
    coefdf.insert(0, 'Degree', polydegs, True)
    coefdf.to_csv(path + 'coefficients_batch_' + str(batchnumber) + '.csv', index=False)
    
    #Adds the noisless signals to a dataframe and saves to a csv file
    pdf = pd.DataFrame(np.array(freud))
    pdf.columns = ['T' + str(i) for i in range(signallength)]
    pdf.insert(0, 'P(x)', ['P_' + str(i) for i in range(numpolys)])
    pdf.insert(0, 'Degree', polydegs, True)
    pdf.to_csv(path + '50k_and_10k_polys_batch_' + str(batchnumber) + '.csv', index=False)
    
    #Note df[i] is the ith polynomial (dataframe type).
    #If you want back the list, pass df[i].tolist().


##################### Training Data Creation ##################################
#Creates the number of training batches that you want. Before generating
#training data in this manner, I had already made 25 batches of training data,
#so here I made 10 batches, labeled 26 through 35

batch_start = 26 #starts with this batch
batch_stop = 36 #stops prior to this batch
batch_number = batch_start
for i in range(batch_stop-batch_start):
    print("starting batch " + str(batch_number))
    batch_maker(batch_number)
    print("batch complete")
    batch_number = batch_number + 1














