##################################################
# This file contains the BayesOptWeb             #
# subclass for the BayesOpt module, as well as a #
# pool manager for concurrency in web apps.      #
#                                                #
# The objective of this class is to provide a    #
# web interface that uses the BayesOptAsync      #
# subclass capabilities and the Flask library    #
# to render a view of an objective function      #
# evaluation and an input for the given result   #
# in a web browser.                              #
#                                                #
# A toy example is provided as the main program  #
#                                                #
# Author: Miguel Marcos (m.marcos@unizar.es)     #
##################################################

from bayesopt_async import BayesOptAsynchronous
from flask import Flask, redirect, url_for, request, session, render_template
from threading import BoundedSemaphore
from datetime import datetime

import numpy as np

class BayesOptWeb(BayesOptAsynchronous):

    # We require a template (as a name string) to render the view.
    # A Flask web application should call the make_query function,
    # which will make the template filled with the appropriate
    # representation of the query accesible with 'get_query'.

    def __init__(self,n,template):

        super().__init__(n)

        # Flask looks for templates in a 'templates' folder
        # Enter the name as a relative path to that folder.
        self.query_template = template

    # For demonstration purposes, we replace the 'print' call of
    # BayesOptAsynchronous with plain text in an HTML page.
    # For more info on Flask templates, check 
    # https://www.tutorialspoint.com/flask/flask_templates.htm

    def make_query(self,query):
        self.query = render_template(self.query_template, 
                                     query=str(query))
        
    # Flask messes with the application context
    # The 'with' statement prevents crash scenarios.
    def targetFunction(self):
        global bo_app
        with bo_app.app_context():
            super().targetFunction()

    # Numpy arrays are not json serializable
    def get_results(self):
        self.final_result_ready.wait()
        return {'mvalue':self.mvalue, 
                'x_out':self.x_out.tolist(), 
                'error':self.error}

class BayesOptWebPool():

    # The BayesOptWebPool class is a dictionary of
    # simultaneous BayesOpt experiments with potentially different
    # problem dimensions and HTML templates. So far it only supports
    # BayesOptWeb instances, so any special features should be
    # implemented on the front end or in subclasses.

    # DISCLAIMER: This class assumes polite behavior from users 
    # (i.e. no unfinished experiments) and has only basic security and 
    # resilience measures against errors. Do not use on production or
    # public networks.

    def __init__(self, max_exp = 10):

        self.max_exp = max_exp
        self.pool = {}
        self.next_id = 0
        self.id_semaphore = BoundedSemaphore()

    # Adds a new experiment to the pool. Returns the ID of the new
    # experiment (which may not be the provided one, if any was) and
    # an error flag (which is set to False on success)
    def new_exp(self,n,template,id=None):

        self.id_semaphore.acquire()
        if len(self.pool) < self.max_exp:
            if id is None:
                id = ""
            id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + \
                    str(self.next_id) + str(id)
            self.next_id += 1
            self.pool[id] = BayesOptWeb(n,template)
            self.id_semaphore.release()
            return id, False
        else:
            self.id_semaphore.release()
            return -1, True

    # Sets the parameters of the experiment with the given ID, if there
    # is one. Returns whether this was True or not.
    def set_exp_params(self,id,params,lb,ub):
        ret = id in self.pool
        if ret:
            self.pool[id].params = params
            self.pool[id].lower_bound = lb
            self.pool[id].upper_bound = ub
        return ret
    
    # Launches the optimization thread of the experiment with the 
    # given ID, if there is one and it hasn't been launched yet. 
    # Returns whether this was True or not.
    def start_exp(self,id):
        ret = id in self.pool
        if ret:
            ret = self.pool[id].optimize()
        return ret
    
    # Waits for an indicated experiment's query to be ready, or returns
    # immediately if there is no experiment with the given ID.
    def wait_for_exp_query(self,id):
        if id in self.pool:
            self.pool[id].wait_for_query()
            return self.pool[id].get_query()
        else:
            return None       
    
    # Sets the result for an experiment's query, if an experiment with
    # the provided ID is in the pool.
    # Returns whether this was True or not.
    def set_exp_query_result(self,id,result):
        ret = id in self.pool
        if ret:
            self.pool[id].set_result(result)
        return ret

    # Waits for an indicated experiment's query result to be registered,
    # or returns immediately if the ID is not in the pool.
    # If there was such an experiment, the function returns whether the
    # experiment is finished or not.
    # Call after setting a query result.
    def wait_for_exp_eval(self,id):
        if id in self.pool:
            return self.pool[id].wait_for_eval()
        else:
            return None           
    
    # Removes the experiment with the given ID from the pool if it
    # exists and it has finished or hasn't started. Returns whether this
    # was True or not, along with the results of the experiment. 
    def remove_exp(self,id):
        ret = (id in self.pool) and (self.pool[id].has_finished() or
                                     not self.pool[id].has_started())
        results = None
        if ret:
            results = self.pool[id].get_results()
            self.pool.pop(id)
        return results, ret

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-

bo_app = Flask(__name__)
bo_app.secret_key = 'everything is Gaussian'
bo_web_pool = BayesOptWebPool()

@bo_app.route('/')
def index():
   return render_template('basic_bo_index.html')

@bo_app.route('/start', methods = ['GET', 'POST'])
def start_exp():

    if request.method == 'GET':
        return redirect(url_for('index'))
    else:

        global bo_web_pool
        
        bo_id, err = bo_web_pool.new_exp(2,'basic_bo_query.html',
                                         id=request.form['name'])
        if err:
            return redirect(url_for('index'))
        else:
            session['bo_id'] = bo_id
            params = {}
            n = 2
            lb = np.zeros((n,)) # Lower bounds
            ub = np.ones((n,)) # Upper bounds
            params['n_init_samples'] = 2
            params['n_iterations'] = 2
            params['load_save_flag'] = 2 # Save
            params['save_filename'] = "exps/"+session['bo_id']+".txt"
            params['verbose_level'] = 5 # Debug -> logfile
            params['log_filename'] = "exps/"+session['bo_id']+"_log.txt"
            bo_web_pool.set_exp_params(bo_id,params,lb,ub)
            bo_web_pool.start_exp(bo_id)
            return redirect(url_for('query'))

@bo_app.route('/query')
def query():
    if 'bo_id' not in session:
        redirect(url_for('index'))
    else:

        global bo_web_pool

        bo_id = session['bo_id']
        query_template = bo_web_pool.wait_for_exp_query(bo_id)
        return query_template
    
@bo_app.route('/input_result', methods=['POST'])
def query_input():
    if 'bo_id' not in session:
        redirect(url_for('index'))
    else:

        global bo_web_pool

        bo_id = session['bo_id']
        q_result = float(request.form['result'])
        bo_web_pool.set_exp_query_result(bo_id,q_result)
        finished = bo_web_pool.wait_for_exp_eval(bo_id)
        if not finished:
            return redirect(url_for('query'))
        else:
            bo_result, ret = bo_web_pool.remove_exp(bo_id)
            session['bo_result'] = bo_result
            return redirect(url_for('end_exp'))
    

@bo_app.route('/result')
def end_exp():
    if ('bo_id' not in session) or ('bo_result' not in session):
        redirect(url_for('index'))
    else:

        global bo_web_pool

        bo_result = session['bo_result']
        return render_template('basic_bo_result.html',res=bo_result)
    
@bo_app.route('/logout',methods=['GET','POST'])
def logout():
    session.pop('bo_id')
    session.pop('bo_result')
    return redirect(url_for('index'))

if __name__ == '__main__':
   bo_app.run(debug = True)
    
