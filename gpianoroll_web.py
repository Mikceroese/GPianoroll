##################################################
# A specific version of BayesOptWeb to create    #
# sustain and render GPianoroll instances        #
#                                                #
# Author: Miguel Marcos (m.marcos@unizar.es)     #
##################################################

import numpy as np
import math

from flask import Flask, redirect, url_for, request, session, render_template, send_from_directory
from datetime import datetime
from os.path import join
from os import getcwd, remove

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

    def __init__(self,name,control_group=False):

        self.n = 10
        self.true_n = 256

        self.control = control_group
        if self.control:
            self.n = self.true_n

        super().__init__(self.n,'gpianoroll_query.html')

        # Default parameters
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

        # Control group parameters
        if self.control:
            params['n_init_samples'] = 63 # +1 for the mid sample

        self.params = params

        lb = np.zeros((self.n,)) # Lower bounds
        ub = np.ones((self.n,)) # Upper bounds
        self.lower_bound = lb
        self.upper_bound = ub

        self.name = 'GPianoroll_'+name
        self.sample_name = 'samples/'+self.name
        self.sample_url = '/samples/'+self.name+'.mid'
        self.best_sample_name = 'samples/'+self.name+'_best'
        self.best_sample_url = '/samples/'+self.name+'_best.mid'

        self.true_n = 256 # True dimension of the model's input
        self.mat_A = np.random.randn(self.n,self.true_n)
        Q , _ = np.linalg.qr(self.mat_A.T)
        self.mat_Q = Q.T

        # Set maximum and minimum scores
        self.min_score = 0.0
        self.max_score = 100.0
        self.best_score = 0
        self.best_result = 1000000000
        self.mid_score = -1.0

        self.current_iter = 1
        self.best_iter = 1

        self.helper_visibility = "visible"

        self.mid_query = self.make_query(np.ones(self.n),first=True)
        self.mid_score = 0.0        

    # Call to set the result and notiying it
    def set_result(self,result):
        self.result = 1- ((result - self.min_score) / (self.max_score-self.min_score))
        if self.result <= self.best_result:
            self.best_score = result
            self.best_result = self.result
            write_sample(self.m,self.best_sample_name)
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
        q = np.matmul(query,self.mat_Q)

        # Clip the resulting query
        q = np.clip(q,-4.0,4.0)

        return q

    def generate_sample(self,query):

        global gen

        if not self.control:
            q = self.uniform_to_normal(query)
            q = self.remap_query(q)
            sample = gen(np_to_tensor(q))
        else:
            sample = gen(np_to_tensor(self.uniform_to_normal(query)))
        note_thresholds = [0.60828567, 0.55597573, 0.54794814]
        sample = clip_samples(sample,note_thresholds)
        return sample

    def make_query(self,query,first=False):
        sample = self.generate_sample(query)
        self.m = samples_to_multitrack(tensor_to_np(sample))
        write_sample(self.m,self.sample_name)
        if first:
            write_sample(self.m,self.best_sample_name)
        self.query = render_template(self.query_template,
                                    mid=self.sample_url,
                                    best_mid=self.best_sample_url,
                                    best_score=int(self.best_score),
                                    helpervisible=self.helper_visibility)
        return self.query

    def targetFunction(self):

        global bo_app
        with bo_app.app_context():

            self.query = self.mid_query
            self.query_ready.set()
            self.wait_for_result()
            self.mid_score = self.result
            self.best_result = self.mid_score
            self.eval_done.set()

            self.mvalue, self.x_out, self.error = BayesOptContinuous.optimize(self)
            # Optimization is not done, just paused
            self.opt_done.clear() 

            if not self.control:

                self.it_semaphore.acquire()
                self.remaining_iterations = 2*self.params['n_init_samples']
                self.it_semaphore.release()
            
                # At this point we have scores for the 21 initial samples
                # plus the middle one.

                bo_dict = BayesOptDict(self.params['save_filename'])
                for i in range(self.params['n_init_samples']):
                    mid_sample = np.round(np.random.uniform(size=self.n),3)
                    mid_sample[np.arange(0,self.n,2)] = 1 - 1e-6
                    bo_dict.add_sample(mid_sample,self.mid_score)

                bo_dict.set_init_samples(self.params['n_init_samples']*2)
                bo_dict.set_num_iter(self.params['n_init_samples'])
                bo_dict.save_txt(self.params['save_filename'])

                # We include 21 points mapped to the center of the latent
                # space during Box-Muller. The user has only graded 22
                # samples, but we have scores for 42.

                self.params['n_init_samples'] *= 2
                self.params['n_iterations'] = self.params['n_init_samples']
                self.params['load_save_flag'] = 3 # 3 - Load and save
                self.params['load_filename'] = self.params['save_filename']

                self.mvalue, self.x_out, self.error = BayesOptContinuous.optimize(self)
                # With 42 iterations, the user will have graded 64 songs

            sample = tensor_to_np(self.generate_sample(self.x_out))
            write_sample(samples_to_multitrack(sample),self.best_sample_name)
            remove(self.sample_name+".mid")           

            self.final_result_ready.set()

    def has_finished(self):
        opt_started = self.params['n_iterations'] > 0
        return self.opt_done.is_set() and opt_started
    
    def get_best_sample_url(self):
        return self.best_sample_url

class GPianorollWebPool(BayesOptWebPool):

    def __init__(self, max_exp = 10):

        super().__init__(max_exp = max_exp)

    def new_exp(self,id=None):

        self.id_semaphore.acquire()
        if len(self.pool) < self.max_exp:
            if id is None:
                id = str(self.next_id) + "_" + \
                     datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
            else:
                id = str(self.next_id) + "_" + str(id) + "_" + \
                     datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
            self.next_id += 1
            self.pool[id] = GPianorollWeb(id,control_group=True)
            self.id_semaphore.release()
            return id, False
        else:
            self.id_semaphore.release()
            return -1, True

    def set_exp_helper_visibility(self, id, visible='visible'):
        if id in self.pool:
            self.pool[id].helper_visibility=visible

    def remove_exp(self, id):
        ret = (id in self.pool) and (self.pool[id].has_finished() or
                                     not self.pool[id].has_started())
        results = None
        url = None
        if ret:
            url = self.pool[id].get_best_sample_url()
            results = self.pool[id].get_results()
            self.pool.pop(id)
        return results, url, ret

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
   return render_template('gpianoroll_index.html')

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
        helper_visible = request.form['helper-state']
        q_result = float(request.form['score'])
        bo_web_pool.set_exp_helper_visibility(bo_id,helper_visible)
        bo_web_pool.set_exp_query_result(bo_id,q_result)
        finished = bo_web_pool.wait_for_exp_eval(bo_id)
        if not finished:
            return redirect(url_for('query'))
        else:
            bo_result, mid_url, ret = bo_web_pool.remove_exp(bo_id)
            session['bo_result'] = bo_result
            session['bo_mid_url'] = mid_url
            return redirect(url_for('end_exp'))
    

@bo_app.route('/result')
def end_exp():
    if ('bo_id' not in session) or ('bo_result' not in session):
        redirect(url_for('index'))
    else:
        mid_url = session['bo_mid_url']
        return render_template('gpianoroll_result.html',mid=mid_url)
    
@bo_app.route('/logout',methods=['GET','POST'])
def logout():
    session.pop('bo_id')
    session.pop('bo_result')
    session.pop('bo_mid_url')
    return redirect(url_for('index'))

if __name__ == '__main__':
   bo_app.run(debug = True,host="0.0.0.0",port=8000)
    
