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

    # We provide an auxiliar thread for optimizing without blocking,
    # a response event to notify when a result has been given,
    # and a variable for said result.
    # A 'query' variable provides access from the outside and the
    # 'query_ready' event can be used for waiting for it.

    def __init__(self,n):

        super().__init__(n)

        self.opt_thread = Thread(group=None,target=super().optimize)

        self.result = None
        self.result_ready = Event()

        self.query = None
        self.query_ready = Event()

    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-
    # Result functions
    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-

    # Call to set the result *before* notiying
    def set_result(self,result):
        self.result = result

    # Call to wake sleeping threads *after* setting the result
    def notify_result(self):
        self.result_ready.set()

    # (Internal function, do not call from outside)
    # Sends the optimization thread to sleep
    def wait_for_result(self):
        self.result_ready.wait()

    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-
    # Query functions
    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-

    # New function for all processing related to the query
    # (e.g. graphic representation, forward pass...)
    # This is the function subclasses should override 
    # instead of 'evaluateSample'
    def make_query(self,query):
        self.query = query
        print(query)

    # Call to ensure query is ready and updated
    def wait_for_query(self):
        self.query_ready.wait()
        self.query_ready.clear()

    # Call to retrieve the query
    def get_query(self):
        return self.query

    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-
    # Optimization functions
    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-

    # Redefinition of the BayesOpt library functions 
    # to account for the asynchronous execution.
    def evaluateSample(self,query):
        self.make_query(query)
        self.query_ready.set()
        self.wait_for_result()
        # The optimization thread is blocked here until 
        # 'notify_result' is called.
        self.result_ready.clear()
        return self.result
    
    # Starts the optimization thread
    def optimize(self):
        self.opt_thread.start()

    # Joins the optimization thread. Call on termination.
    def wait_for_finish(self):
        self.opt_thread.join()

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
        x = float(input()) # Change the retrieval method as needed
        bo.set_result(x)
        bo.notify_response()

    bo.wait_for_finish()

    print("Done")
    
