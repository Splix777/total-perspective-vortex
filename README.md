In progress, check back for updates...

<h1 align="center">Total Perspective Vortex</h1>

<div align="center">
<img src="images/logo.png" alt="Harry Potter" width="35%">
</div>


# Plug your brain to the shell

## Introduction

This subject aims to create a brain computer interface based on electroencephalographic
data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG
reading, you’ll have to infer what he or she is thinking about or doing—(motion) A or B
in a t0 to tn timeframe.

## Objectives

- Process EEG data (parsing and filtering)
- Implement a dimensionality reduction algorithm
- Use the pipeline API of scikit-learn
- Classify a data stream in "real-time"

## General Instructions

You’ll have to process data coming from cerebral activity with machine
learning algorithms. The data was created during a motor imagery experiment,
where people had to do or imagine a hand or feet movement. Those people
were told to think or do a movement corresponding to a symbol displayed
on screen. The results are cerebral signals with labels indicating moments
where the subject had to perform a certain task.

You’ll have to code in Python as it provides MNE, a library specialized
in EEG data processing and, scikit-learn, a library specialized in
machine learning.

The subject focuses on implementing the algorithm of dimensionality reduction,
to further transform filtered data before classification. This algorithm
will have to be integrated within sklearn, so you’ll be able to use sklearn
tools for classification and score validation.

## V.1.1 Preprocessing, Parsing and Formating

First, you’ll need to parse and explore EEG data with MNE, from
[physionet](https://physionet.org/content/eegmmidb/1.0.0/).
You will have to write a script to visualize raw data then filter it to
keep only useful frequency bands, and visualize again after this preprocessing.

This part is where you’ll decide which features you’ll extract from the
signals to feed them to your algorithm. So you’ll have to be thorough picking
what matters for the desired output.

One example is to use the power of the signal by frequency and by channel
to the pipeline’s input.

## V.1.2 Treatment pipeline

Then the processing pipeline has to be set up:

- Dimensionality reduction algorithm (i.e. : PCA, ICA, CSP, CSSP...).
- Classification algorithm, there are plenty of choices among those
available in sklearn, to output the decision of what data chunk
correspond to what kind of motion.
- "Playback" reading on the file to simulate a data stream.

It is advised to first test your program architecture with sklearn and
MNE algorithms, before implementing your own CSP or whatever algorithm 
you chose. The program will have to contain a script for training and a
script for prediction. The script predicting output will have to do
it on a stream of data, and before a delay of 2s, after the data chunk was
sent to the processing pipeline. (you should not use mne-realtime)

You have to use the pipeline object from sklearn (use baseEstimator and
transformerMixin classes of sklearn)

## V.1.3 Implementation

The aim is to implement the dimensionality reduction algorithm. This means
to express the data with the most meaningful features, by determining a 
projection matrix.

This matrix will project the data on a new set of axes that will express
the most "important" variations. It is called a change of basis, and it is
a transformation composed of rotations, translations and scaling operations.

As such, the PCA considers your dataset and determines new basis components,
sorted by how much those axises account for variations in the data.

The [CSP](https://doc.ml.tu-berlin.de/bbci/publications/BlaTomLemKawMue08.pdf) 
or common spatial patterns, analyses the data depending on the output
classes and try to maximize the variations between them.
PCA is a more general algorithm, but CSP is more used in EEG BCIs.

Let's take the formal expression of an EEG signal:

- `N` the number of events per class.
- `ch` number of channels (electrodes)
- `time` the length of event recording

Considering the extracted signal matrix X ∈ R d∗N, knowing that d = ch ∗ time 
is the dimension of a signal vector for an event record.

## V.1.4 Train, Validation and Test

- You have to use cross_val_score on the whole processing pipeline, to
evaluate your classification.

- You must choose how to split your dataset between Train, Validation
and Test set (Do not overfit, with different splits each time)

- You must have 60% mean accuracy on all subjects used in your Test Data
(corresponding to the six types of experiment runs and on never-learned data)

- You can train / predict on the subject and the task of your choice


