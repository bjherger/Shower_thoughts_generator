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
