In progress, check back for updates...

<h1 align="center">Total Perspective Vortex</h1>

<div align="center">
<img src="images/logo.png" alt="Harry Potter" width="35%">
</div>


# Plug your brain to the shell

## Introduction

This subject aims to create a brain computer interface based on electroencephalographic
data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG
reading, you’ll have to infer what he or she is thinking about or doing - (motion) A or B
in a t0 to tn timeframe.

## Objectives

- Process EEG datas (parsing and filtering)
- Implement a dimensionality reduction algorithm
- Use the pipeline API of scikit-learn
- Classify a data stream in "real-time"

## General Instructions

You’ll have to process data coming from cerebral activity, with machine
learning algorithms. The data was mesured during a motor imagery experiment,
where people had to do or imagine a hand or feet movement. Those people
were told to think or do a movement corresponding to a symbol displayed
on screen. The results are cerebral signals with labels indicating moments
where the subject had to perform a certain task.

You’ll have to code in Python as it provides MNE, a library specialized
in EEG data processing and, scikit-learn, a library specialized in
machine learning.

The subject focuses on implementing the algorithm of dimensionality reduction,
to further transform filtered data before classification. This algorithm
will have to be integrated within sklearn so you’ll be able to use sklearn
tools for classification and score validation.

## V.1.1 Preprocessing, Parsing and Formating

### I.1.1.1 Preprocessing

First, you’ll need to parse and explore EEG data with MNE, from
[physionet](https://physionet.org/content/eegmmidb/1.0.0/).
You will have to write a script to visualize raw data then filter it to
keep only useful frequency bands, and visualize again after this preprocessing.

This part is where you’ll decide which features you’ll extract from the
signals to feed them to your algorithm. So you’ll have to be thorough picking
what matters for the desired output.

One example is to use the power of the signal by frequency and by channel
to the pipeline’s input.

[plots](images/plots/plots.html)















