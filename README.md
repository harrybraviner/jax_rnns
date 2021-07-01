JAX RNNs
========

Aim of this repo is to test out patterns for writing RNNs in JAX.

# Atomic Tests

RNNs are often used in language or time series modelling applications in which performance is difficult.
To test that RNNs are satisfying certain primitive properties, we need some simpler _atomic tests_.

## Correct injection of hidden state `h0`

To reduce operations to matrix multiplications, RNNs usually require padding of all but the longest sequence in a batch.
For the padded sequences, the initial hidden state should be injected at the first non-padded state.
However, an easy mistake would be to fail to do this, and just inject `h0` at the first (padded) state.

To test for this, we construct a dataset consisting of uniformly chosen sequence length between 1 and 16.
The expected output at the end of the sequence is 1 for a sequence of length <= 8, and 0 otherwise.
Inputs are always a vector of zeros at each timestep.

## Remembering an impulse

We take sequences of length 16, and a two-element input vector, [x0, x1].
Usually [x0=1, x1=0], with a small number of [x0=0, x1=1] inputs.
If such an x1=1 input occurs within 8 steps of the end of the RNN, the ground-truth is 1, else it is 0.

This tests that ability to remember an impulse, and then forget it after some time.

