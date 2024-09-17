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

import numpy as np
import math

from flask import Flask, redirect, url_for, request, session, render_template, send_from_directory
from datetime import datetime
from os.path import join
from os import getcwd

from bayesopt_web import BayesOptWeb, BayesOptWebPool
from bayesoptmodule import BayesOptContinuous
from bopt_dict import BayesOptDict

from musegan import Generator # The model is defined here
from musegan import clip_samples, samples_to_multitrack, write_sample
from utils import *

class GPianorollWeb(BayesOptWeb):

    # We require a template (as a name string) to render the view.
    # A Flask web application should call the make_query function,
    # which will make the template filled with the appropriate
    # representation of the query accesible with 'get_query'.

    def __init__(self,name):

        self.n = 10
        self.true_n = 256

        super().__init__(self.n,'gpianoroll_query.html')

        params = {}
        params['n_init_samples'] = 21
        params['init_method'] = 2 # Sobol
        params['noise'] = 1e-2 # Default 1e-6
        params['n_iterations'] = 0
        params['n_iter_relearn'] = 1
        params['l_type'] = 'mcmc'
        params['load_save_flag'] = 2 # 2 - Save
        params['save_filename'] = "exps/"+name+".txt"
        params['verbose_level'] = 5 # Debug -> logfile
        params['log_filename'] = "exps/"+name+"_log.txt"

        self.params = params

        lb = np.zeros((self.n,)) # Lower bounds
        ub = np.ones((self.n,)) # Upper bounds
        self.lower_bound = lb
        self.upper_bound = ub

        self.name = 'GPianoroll_'+name
        self.sample_name = 'samples/'+self.name
        self.sample_url = '/samples/'+self.name+'.mid'

        self.true_n = 256 # True dimension of the model's input
        self.mat_A = np.random.randn(self.n,self.true_n)
        # Precalculate variances for remapping
        # (See 'remap_query' below)
        self.vars = np.sum(np.square(self.mat_A),axis=0)
        self.stds = np.sqrt(self.vars)

        # Set maximum and minimum scores
        self.min_score = 0.0
        self.max_score = 100.0
        self.best_score = -1.0
        self.mid_score = -1.0

        self.current_iter = 1
        self.best_iter = 1

        self.mid_query = render_template('gpianoroll_query.html',
                                         mid='/static/mid_sample.mid')
        self.mid_score = 0.0        

    # Call to set the result and notiying it
    def set_result(self,result):
        self.result = 1- ((result - self.min_score) / self.max_score)
        if self.result < self.best_score:
            self.best_score = self.result
        self.result_ready.set()

    def uniform_to_normal(self,query):

        # Box-Muller transform
        even = np.arange(0,query.shape[-1],2)
        q_even = query[even]
        q_even[q_even==0] += 1e-6
        Rs = np.sqrt(-2*np.log(q_even))
        thetas = 2*math.pi*(query[even+1])
        cos = np.cos(thetas)
        sin = np.sin(thetas)
        query = np.stack([Rs*cos,Rs*sin],-1).flatten()

        return query

    def remap_query(self,query):

        # Map query to the high-dimensional space
        q = np.matmul(query,self.mat_A)

        # Queries are mapped to a ~N(0,1) distribution
        # using the Box-Muller transform.
        # Therefore, the vector-matrix product (q)
        # results in a vector of sums of Normal distributions,
        # which are Normal themselves.

        # Each component i of the resulting vector will follow a Normal
        # distribution with mean 0 and variance V[i] equal
        # to the sum of the squares of the i-th column of mat_A.

        # Thus, we can remap each of this components to ~N(0,1)
        # using the precalculated standard deviations:

        q = np.divide(q,self.stds)

        # Clip the resulting query
        q = np.clip(q,-4.0,4.0)

        return q

    def generate_sample(self,query):

        global gen

        q = self.uniform_to_normal(query)
        q = self.remap_query(q)
        sample = gen(np_to_tensor(q))
        note_thresholds = [0.60828567, 0.55597573, 0.54794814]
        sample = clip_samples(sample,note_thresholds)
        return sample

    def make_query(self,query):
        sample = self.generate_sample(query)
        m = samples_to_multitrack(tensor_to_np(sample))
        write_sample(m,self.sample_name)
        self.query = render_template(self.query_template,
                                     mid=self.sample_url)
        return self.query

    def targetFunction(self):

        global bo_app
        with bo_app.app_context():

            self.query = self.mid_query
            self.query_ready.set()
            self.wait_for_result()
            self.mid_score = self.result
            self.best_score = self.mid_score

            self.mvalue, self.x_out, self.error = BayesOptContinuous.optimize(self)
            # Optimization is not done, just paused
            self.opt_done.clear() 

            self.it_semaphore.acquire()
            self.remaining_iterations = 42
            self.it_semaphore.release()
           
            # At this point we have scores for the 21 initial samples
            # plus the middle one.

            bo_dict = BayesOptDict(self.params['save_filename'])
            for i in range(21):
                mid_sample = np.round(np.random.uniform(size=self.n),3)
                mid_sample[np.arange(0,self.n,2)] = 1 - 1e-6
                bo_dict.add_sample(mid_sample,self.mid_score)

            bo_dict.set_init_samples(self.params['n_init_samples']+21)
            bo_dict.set_num_iter(self.params['n_init_samples']+21)
            bo_dict.save_txt(self.params['save_filename'])

            # We include 21 points mapped to the center of the latent
            # space during Box-Muller. The user has only graded 22
            # samples, but we have scores for 42.

            self.params['n_init_samples'] += 21
            self.params['n_iterations'] = 3
            self.params['load_save_flag'] = 3 # 3 - Load and save
            self.params['load_filename'] = self.params['save_filename']

            self.mvalue, self.x_out, self.error = BayesOptContinuous.optimize(self)
            # With 42 iterations, the user will have graded 64 songs
            self.final_result_ready.set()

    def has_finished(self):
        opt_started = self.params['n_iterations'] > 0
        return self.opt_done.is_set() and opt_started

class GPianorollWebPool(BayesOptWebPool):

    def __init__(self, max_exp = 10):

        super().__init__(max_exp = max_exp)

    def new_exp(self,id=None):

        self.id_semaphore.acquire()
        if len(self.pool) < self.max_exp:
            if id is None:
                id = str(self.next_id) + \
                     datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
            else:
                id = str(self.next_id) + str(id) + \
                     datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
            self.next_id += 1
            self.pool[id] = GPianorollWeb(id)
            self.id_semaphore.release()
            return id, False
        else:
            self.id_semaphore.release()
            return -1, True

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-

bo_app = Flask(__name__)
bo_app.secret_key = 'everything is Gaussian'
bo_web_pool = GPianorollWebPool()

wd = getcwd() # Working directory
cp_path_gen = join(wd,'museGANgen_DBG_chroma_256_25k.pt') # Path to the generator checkpoint
gen = Generator()
load_checkpoint(cp_path_gen,gen)
if torch.cuda.is_available():
    gen = gen.cuda()
gen.eval()

@bo_app.route('/')
def index():
   return render_template('basic_bo_index.html')

@bo_app.route('/start', methods = ['GET', 'POST'])
def start_exp():

    if request.method == 'GET':
        return redirect(url_for('index'))
    else:

        global bo_web_pool
        
        bo_id, err = bo_web_pool.new_exp(id=request.form['name'])
        if err:
            return redirect(url_for('index'))
        else:
            session['bo_id'] = bo_id
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

@bo_app.route('/samples/<path:path>')
def get_sample(path):
    return send_from_directory('samples', path)

    
@bo_app.route('/input_result', methods=['POST'])
def query_input():
    if 'bo_id' not in session:
        redirect(url_for('index'))
    else:

        global bo_web_pool

        bo_id = session['bo_id']
        q_result = float(request.form['score'])
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
    
