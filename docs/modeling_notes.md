# Modeling Notes

## 2017-12-21

### Callbacks

TensorBoard callback

 - Looking up for [keras docs](https://keras.io/callbacks/#tensorboard)
 - Created callback
 - Looking up how to use tensorboard. Reading [instructions](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
 - Reading [fizzy logic blog post](https://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/)
 - Callback seems to be working, able to check w/ tensorboard
 - To set up tensorboard: call `tensorboard --logdir=~/.logs`, then `open http://localhost:6006`
 
Model check-pointing

 - Reading [keras docs](https://keras.io/callbacks/#modelcheckpoint)
 - Adding model checkpoint callback
 - Installing `h5py`, required to save models
 - Setting up filename, adding to confs
 - Test run
 
Sentence generation

 - Changing sentence generation to callback, so that I can use regular epoch generation
 - Reading [keras docs](https://keras.io/callbacks/#create-a-callback)
 - I'll create a custom class, so that I can store output
 - First pass at creating callback
 - Refactoring calls to other files
 
## 2017-12-22

Ran some hefty models yesterday. Adding h5py file to repo

### Infer

 - Setting up inference pipeline, to complete stubs
 - Creating interface
 - Writing model and data load
 - Starting on transform
 - Updating x y generator to include start and end of phrase brackets
 - Generating observations
 - Writing everything to file
 
### Model2.0

Backlog 

 - Include all printable characters
 - Include `>>>>` at end of post
 - More LSTM nodes
 - LR restarts?
 
Extract changes
 - Including all printable chars
 - Added padding w/ end character
 - Doubled number of LSTM nodes
 
LR restarts

 - Looks like Brad's already worked on this. [repo](https://github.com/bckenstler/CLR)
 - Trying Brad's CLR
 - There seems to be an issue w/ the embedding. Removing CLR to see if that's causing it. 
 - Code still does not run
 - Issue seems to be related to updated list of legal characters. 
 - Modifying model to instead use length of legal characters directly, rather than max value seen in X
 - Re-introducing CLR
 
Data

 - Downloading a 400 day data sample
 
## 2017-12-26

### GIF

Looking up methods for creating gifs

 - This [so](https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python) points to imageio

Single images

 - [ImageFont](http://effbot.org/imagingbook/imagefont.htm), part of Pillow
 - No great options for changing font on same line. Using same font and Pillow instead
 

