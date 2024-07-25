"""
Neuroevolution Module

Implements neuroevolution techniques for optimizing neural network architectures
using genetic programming. It uses the DEAP library for evolutionary algorithms and Keras
for neural network implementation.

Main Components:
- Primitive set creation for neural network layers
- Evaluation functions for symbolic regression
- Toolbox generation for genetic programming
- Utility functions for model creation and data generation

Dependencies:
- deap
- numpy
- tensorflow
- keras

"""

from deap import algorithms, base, creator, tools, gp
import numpy
import operator
import math
from tensorflow import Tensor
import keras.layers as kl
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
import random

# Custom classes for type hinting
class dropout_parm: pass
class kernel_size: pass
class layer_width: pass
class activation_function: pass
class Data: pass
class col_num: pass

def conv1d(layers, width, kernel):
    """Wrapper for Keras.layers.Conv1D that allows parameter passing."""
    return kl.Conv1D(width, kernel_size=kernel)(layers)

def lstm(layers, width):
    """Wrapper for Keras.layers.LSTM that allows parameter passing."""
    return kl.LSTM(width)(layers)

def conv2d(layers, width, kernel):
    """Wrapper for Keras.layers.Conv2D that allows parameter passing."""
    return kl.Conv2D(width, kernel_size=(kernel, kernel))(layers)

def gru(layers, width):
    """Wrapper for Keras.layers.GRU that allows parameter passing."""
    layers2 = layers
    return kl.GRU(width)(layers2)

def dense(layers, width):
    """Wrapper for Keras.layers.Dense that allows parameter passing."""
    layers2 = layers
    return kl.Dense(width)(layers2)

def dropout(layers, parm):
    """Wrapper for Keras.layers.Dropout that allows parameter passing."""
    return kl.Dropout(parm)(layers)

def gaussian(layers, parm):
    """Wrapper for Keras.layers.GaussianNoise that allows parameter passing."""
    return kl.GaussianNoise(parm)(layers)

def concatenate(layers1, layers2):
    """Wrapper for Keras.layers.Concatenate that allows parameter passing."""
    res1 = layers1
    res2 = layers2
    if layers1.shape.ndims > layers2.shape.ndims:
        res1 = kl.Reshape((-1,))(layers1)   
    if layers2.shape.ndims > layers1.shape.ndims:
        res2 = kl.Reshape((-1,))(layers2)   
    return kl.Concatenate()([res1, res2])

def maxPooling1D(layers, kernel):
    """Wrapper for Keras.layers.MaxPooling1D that allows parameter passing."""
    return kl.MaxPooling1D(kernel)(layers)

def maxPooling2D(layers, kernel):
    """Wrapper for Keras.layers.MaxPooling2D that allows parameter passing."""
    return kl.MaxPooling2D((kernel, kernel))(layers)

def activation(layers, activation_function):
    """Wrapper for Keras.layers.Activation that allows parameter passing."""
    return kl.Activation(activation_function)(layers)

def makeInput(data, col_num):
    """Create an input layer based on the data and column number."""
    return kl.Input(input_shape=list(data.values())[col_num])

def extend(f):
    """Identity function needed due to a glitch in DEAP."""
    return f

def createPset():
    """
    Create all of the primitive and ephemeral assignments
    that make up the pset that defines neuroevolution.
    
    Returns:
        gp.PrimitiveSetTyped: A complete primitive set for genetic programming
    """
    pset = gp.PrimitiveSetTyped('main', [Data], Tensor)
    
    # Add primitives
    pset.addPrimitive(makeInput, [Data, col_num], Tensor, name='Input')    
    pset.addPrimitive(conv2d, [Tensor, layer_width, kernel_size], Tensor, name='Conv2D')
    pset.addPrimitive(conv1d, [Tensor, layer_width, kernel_size], Tensor, name='Conv1D')
    pset.addPrimitive(gru, [Tensor, layer_width], Tensor, name='GRU')
    pset.addPrimitive(maxPooling1D, [Tensor, kernel_size], Tensor, name='MaxPool1D')
    pset.addPrimitive(maxPooling2D, [Tensor, kernel_size], Tensor, name='MaxPool2D')
    pset.addPrimitive(dropout, [Tensor, dropout_parm], Tensor, name='Dropout')
    pset.addPrimitive(dense, [Tensor, layer_width], Tensor, name='Dense')
    pset.addPrimitive(activation, [Tensor, activation_function], Tensor, name='Activation')
    pset.addPrimitive(lstm, [Tensor, layer_width], Tensor, name='LSTM')
    pset.addPrimitive(concatenate, [Tensor, Tensor], Tensor, name='Concatenate')
    pset.addPrimitive(kl.BatchNormalization(), [Tensor], Tensor, name='BatchNorm')
    pset.addPrimitive(kl.GlobalAveragePooling2D(), [Tensor], Tensor, name='GlobalAvgPool2D')
    pset.addPrimitive(kl.GlobalAveragePooling1D(), [Tensor], Tensor, name='GlobalAvgPool1D')    
    pset.addPrimitive(extend, [dropout_parm], dropout_parm)
    pset.addPrimitive(extend, [layer_width], layer_width)
    pset.addPrimitive(extend, [kernel_size], kernel_size)
    pset.addPrimitive(extend, [activation_function], activation_function)
    pset.addPrimitive(extend, [col_num], col_num)
    pset.addPrimitive(extend, [Tensor], Tensor)
    pset.addPrimitive(extend, [Data], Data)
    pset.addPrimitive(kl.Embedding(600, 50, trainable=True), [Tensor], Tensor, name='Embedding')
    
    # Add ephemeral constants
    pset.addEphemeralConstant("dropout_parm", lambda: numpy.random.randint(1, 14) * .05, dropout_parm)
    pset.addEphemeralConstant("kernel_size", lambda: numpy.random.randint(2, 4), kernel_size)
    pset.addEphemeralConstant("layer_width", lambda: numpy.random.randint(1, 16) * 16, layer_width)
    pset.addEphemeralConstant("activation_function", lambda: random.choice(('softmax', 'softplus', 'softsign', 'relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear')), activation_function)
    pset.addEphemeralConstant("col_num", lambda: numpy.random.randint(0, 2), col_num)
    
    return pset

def evalSymbReg4(individual, input, output, pre, toolbox, defaultFitness=9999, loss='mae', metrics=['mae'],
                 callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')],
                 optimizer=Adam(lr=0.05, decay=0.001)):
    """
    Evaluation function for symbolic regression individuals.
    
    Args:
        individual: The individual to be evaluated
        input: Input layer of the neural network
        output: Output layer of the neural network
        pre: Preprocessed input data
        toolbox: DEAP toolbox
        defaultFitness: Default fitness value in case of exceptions
        loss: Loss function for model compilation
        metrics: Metrics for model evaluation
        callbacks: Callbacks for model training
        optimizer: Optimizer for model compilation
    
    Returns:
        tuple: Fitness value of the individual
    """
    func = toolbox.compile(expr=individual)
    try:
        print('Begin new candidate.')
        print('Candidate structure: ' + str(individual))
        layers = func(pre)
        
        if K.ndim(layers) > 2:
            print('Flattening layers...')
            flat = kl.Flatten()(layers)
            print('Flattening complete.')
            output2 = output()(flat)
        else:
            print('Skipping flattening layers...')
            output2 = output()(layers)
        
        model = Model(input, output2)
        print(model.summary())
        batch_size = 2000
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        model.fit_generator(dictGenerator(X_train, Y_train, batch_size),
                            steps_per_epoch=10,
                            epochs=20, callbacks=callbacks, verbose=1,
                            validation_data=dictGenerator(X_train, Y_train, batch_size),
                            validation_steps=5)
        
        return (min(model.history.history['val_loss']) * .7 + model.history.history['val_loss'][-1] * .25 + 
                min(model.history.history['loss']) * .05),
    except Exception as e:
        print('Exception!')
        print(e)
        return defaultFitness,

def generateToolbox(input, output, pre=None, defaultFitness=9999, loss='mae',
                    metrics=['mae'], callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')], pset=None):
    """
    Generate a DEAP toolbox for evolutionary algorithms.
    
    Args:
        input: Input layer of the neural network
        output: Output layer of the neural network
        pre: Preprocessed input data (default: None)
        defaultFitness: Default fitness value (default: 9999)
        loss: Loss function for model compilation (default: 'mae')
        metrics: Metrics for model evaluation (default: ['mae'])
        callbacks: Callbacks for model training
        pset: Primitive set (default: None)
    
    Returns:
        base.Toolbox: DEAP toolbox for evolutionary algorithms
    """
    if pre is None:
        pre = input
    if pset is None:
        pset = createPset()
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    from functools import partial
    evalSymbReg3 = partial(evalSymbReg4, input=input, output=output, pre=pre, toolbox=toolbox, 
                           defaultFitness=defaultFitness, loss=loss, metrics=metrics, callbacks=callbacks)
    
    toolbox.register("evaluate", evalSymbReg3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=25))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=25))
    
    return toolbox

def getStats():
    """
    Generate standard statistics object for logging.
    
    Returns:
        tools.MultiStatistics: Statistics object for logging
    """
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    return mstats

def dictGenerator(features, labels, batch_size):
    """
    A generator that sends batches of random samples from the source data.
    
    Args:
        features (dict): A dictionary containing entries that are features of the same length
        labels (list or numpy.array): A list or numpy array of output labels
        batch_size (int): The batch size to return
    
    Yields:
        tuple: A tuple consisting of the dictionary sample and corresponding labels
    """
    length = len(next(iter(features.values())))
    import random
    batch_features = features.copy()
    while True:
        index = random.sample(range(length), batch_size)
        for key, value in features.items():
            batch_features[key] = value[index]
        yield batch_features, labels[index]

def modelizer(toolbox, individual, input, pre, output):
    """
    Create a Keras model from a population individual.
    
    Args:
        toolbox: Toolbox that originally defined the population
        individual: Individual from the population to be compiled
        input: Input layer
        pre: Preprocessed input layer
        output: Output layer
    
    Returns:
        keras.models.Model: A Keras model
    """
    func = toolbox.compile(individual)
    layers = func(pre)
    
    if K.ndim(layers) > 2:
        print('Flattening layers...')
        flat = kl.Flatten()(layers)
        outputLayer = output()(flat)
    else:
        print('Skipping flattening layers...')
        outputLayer = output()(layers)
    
    model = Model(input, outputLayer)
    return model

def picklePopulation(pop, hof, log, filename):
    """
    Pickle the population, hall of fame, and logbook.
    
    Args:
        pop: Population to pickle
        hof: Hall of fame to pickle
        log: Logbook to pickle
        filename (str): Base filename for the pickle file
    
    Returns:
        bool: True if pickling was successful
    """
    cp = dict(population=pop, halloffame=hof, logbook=log)
    import pickle
    import datetime
    nw = datetime.datetime.now().strftime('%Y%m%d%H%M%S')