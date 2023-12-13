# GPianoroll

Music is a mix of science and humanism. A musical piece's impression depends on the physical properties of the sounds that make it, but also on the listener. In this project, we propose an evaluation method that leverages human feedback as our quality measurement for music. We perform Bayesian Optimization over a low-dimensional latent space from where we take samples, which are later mapped to the high-dimensional latent space of the musical score generator MuseGAN, and ask the final user to evaluate how they like the output of the model. At the end of the process, our method finds which point of the latent space is more to the subject's liking, and extracts one final sample. 

# Files

Our work includes the following files:
 - **A notebook for training our baseline**, which is a custom version of the **MuseGAN** model, built with PyTorch.
 - **A pretrained version** of the baseline, result of running the notebook.
 -  **Code for running experiments**, which include the description of the model, a rudimentary GUI, and auxiliary code for loading the model and running the experiments. 


# Running the code
