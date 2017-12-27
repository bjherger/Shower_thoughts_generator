# Shower Thought Generator

**tl;dr:** I tried to train a Deep Learning character model to have shower thoughts, using Reddit data. Instead it 
learned pithiness, curse words and clickbait-ing.

A character level RNN / LSTM model, trained on data pulled from Reddit.    

## Quick start
  
To run the Python code, complete the following:
```bash

# Create python virtual environment. Assumes that Anaconds is installed. If not, see Python Enivronment section of 
# README
conda env create -f environment.yml 

# Activate python virtual environment
source activate shower

# Run script
cd bin/
python main.py

# Warning: This will train a model with a large existing data pull. Depending on your setup, this could take hours to 
# months.   
```

## Getting started

### Repo structure

 - `bin/main.py`: Entry point into the Reddit scraper / model training code
 - `conf/conf.yaml.template`: Template for run configs. See [confs](#confs)
 - `bin/generator.py`: Entry point into code that will generate blog posts, given user seeds
 - `bin/post_viz.py`: Entry point into code that will generate gifs from model output. 

### Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the 
appropriate packages, please install [Anaconda](https://www.continuum.io/downloads), and install the environment 
described in environment.yml (Instructions [here](http://conda.pydata.org/docs/using/envs.html), under *Use 
environment from file*, and *Change environments (activate/deactivate)*). 


### Confs

This application requires some configuration before it can run correctly. Please use the commands below to set up the 
configs:

```bash
# Create configuration file
cp conf/confs.yaml.template conf/confs.yaml

# Fill out confs (This requires work on your end!)
open conf/confs.yaml
```

## Contact
Feel free to contact me at 13herger `<at>` gmail `<dot>` com
