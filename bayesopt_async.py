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
# A toy example is provided as the main program #
#                                               #
# Author: Miguel Marcos (m.marcos@unizar.es)    #
#################################################

from bayesoptmodule import BayesOptContinuous
from threading import Thread, Event, BoundedSemaphore

import numpy as np

class BayesOptAsynchronous(BayesOptContinuous):

    def __init__(self,n):

        super().__init__(n)

        # Storing and notifying a new query
        self.query = None
        self.query_ready = Event()

        # Storing and notifying query evaluations
        self.result = None
        self.result_ready = Event()
        self.eval_done = Event()

        # Handling progress and termination
        self.it_semaphore = BoundedSemaphore()
        self.remaining_iterations = -1
        self.opt_done = Event()

        # Variables for storing the final optimization result
        self.final_result = {}
        self.mvalue = None
        self.x_out = None
        self.error = False
        self.final_result_ready = Event()

        # Background thread, for non-blocking execution
        self.opt_thread = Thread(group=None,target=self.targetFunction)

    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-
    # Query result functions
    # -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-

    # Call to set the result and notiying it
    def set_result(self,result):
        self.result = result
        self.result_ready.set()

    # (Internal function, do not call from outside)
    # Sends the optimization thread to sleep
    def wait_for_result(self):
        self.result_ready.wait()
        self.result_ready.clear()

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
        # 'set_result' is called.

        self.it_semaphore.acquire()
        self.remaining_iterations -= 1
        if self.remaining_iterations == 0:
            self.opt_done.set()
        self.it_semaphore.release()

        self.eval_done.set()

        return self.result
    
    # Call this to ensure the number of remaining iterations and 
    # the termination flag have been updated after a query.
    # Returns whether this was the last evaluation or not.
    def wait_for_eval(self):
        self.eval_done.wait()
        self.eval_done.clear()
        return self.has_finished()
    
    # Proxy function for the optimization thread
    # (Threads can't return values, but values stored inside are saved)
    def targetFunction(self):
        self.mvalue, self.x_out, self.error = super().optimize()
        self.final_result_ready.set()
    
    # Starts the optimization thread (only once)
    def optimize(self):
        start = not self.has_started()
        if start:
            self.it_semaphore.acquire()
            self.remaining_iterations = self.params['n_init_samples'] + \
                                        self.params['n_iterations']
            self.it_semaphore.release()
            self.opt_thread.start()
        return start

    # Joins the optimization thread. Call on termination.
    def wait_for_finish(self):
        self.opt_thread.join()

    # If the number of iterations is not -1, 'optimize' was called.
    def has_started(self):
        self.it_semaphore.acquire()
        started = self.remaining_iterations >= 0
        self.it_semaphore.release()
        return started

    # Check for termination flag
    def has_finished(self):
        return self.opt_done.is_set()
    
    # Return a dictionary with the result variables
    def get_results(self):
        self.final_result_ready.wait()
        return {'mvalue':self.mvalue, 
                'x_out':self.x_out, 
                'error':self.error}

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

    while True:
        x = float(input()) # Change the input method as needed
        bo.set_result(x)
        finish = bo.wait_for_eval()
        if finish:
            break

    bo.wait_for_finish()

    print("Done")
    print(bo.get_results())
    
