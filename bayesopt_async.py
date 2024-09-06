#################################################
# This file contains the BayesOptAsynchronous   #
# subclass for the BayesOpt module              #
#                                               #
# The objective of this class is to provide a   #
# comfortable way of managing long wait tasks   #
# during evaluations in Bayesian Optimization   #
# (e.g. I/O operations)                         #
#                                               #
# Subclasses of BayesOptAsynchronous can        #
# use its functionalities just by overriding    #
# the 'make_query' method.                      #
#                                               #
# Author: Miguel Marcos (m.marcos@unizar.es)    #
#################################################

from bayesoptmodule import BayesOptContinuous
from threading import Thread, Event

import numpy as np

class BayesOptAsynchronous(BayesOptContinuous):

    # We require an auxiliar thread for optimizing without blocking,
    # a response event to notify when a result has been given,
    # and a variable for said result.

    def __init__(self,n):

        super().__init__(n)

        self.opt_thread = Thread(group=None,target=super().optimize)
        self.response = Event()
        self.result = None

    def set_result(self,result):
        self.result = result

    def notify_response(self):
        self.response.set()

    def wait_for_response(self):
        self.response.wait()

    def wait_for_finish(self):
        self.opt_thread.join()

    # We make a new function for all processing related to the query
    # (e.g. graphic representation, forward pass...)
    # This is the function subclasses should override instead of 'evaluateSample'

    def make_query(self,query):
        print(query)

    # Lastly, we redefine the BayesOpt library functions 
    # to account for the asynchronous execution.

    def evaluateSample(self,query):
        self.make_query(query)
        self.wait_for_response()
        self.response.clear()
        return self.result
    
    def optimize(self):
        self.opt_thread.start()

if __name__ == "__main__":

    params = {}
    params['n_init_samples'] = 2
    params['n_iterations'] = 2

    n = 2
    lb = np.zeros((n,)) # Lower bounds
    ub = np.ones((n,)) # Upper bounds

    bo = BayesOptAsynchronous(n)
    bo.params = params
    bo.lower_bound = lb 
    bo.upper_bound = ub

    bo.optimize()

    for i in range(params['n_init_samples']+params['n_iterations']):
        x = float(input()) # Change the result retrieval method as needed
        bo.set_result(x)
        bo.notify_response()

    bo.wait_for_finish()

    print("Done")
    
