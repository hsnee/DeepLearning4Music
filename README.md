# DeepLearning4Music

Training a Long Short-Term Memory network to generate music by learning the recurrent structure of music. This was a project with Yizhou He and Peter Schaldenbrand at Carnegie Mellon University.

## Dataset
We compiled our own dataset for use in this project. The data is MIDI files collected online from sources under licenses allowing free usage for research purposes.
The data we collected has reached over 100,000 songs, and has been pushed on to github [here](https://github.com/hsnee/DeepLearning4Music/tree/master/data) mostly organized by genre. We use the python library [mido](https://mido.readthedocs.io/en/latest/index.html) to read in the MIDI files

## Network Architecture

We used `keras` to implement the following network architecture:



Layer (type)   | Output shape                           | Parameter Number
------- | ------------------------------------- | --------
lstm_1 (LSTM)  | (None, 48)           | 9984
dense_1 (Dense) | (None, 24)   | 1176
dense_2 (Dense)   | (None, 3)      | 75

## Loss Function
Most typical loss functions used in machine learning reward close values and penalize far values from the truth. This is a meaningful approach to take with velocity and time deltas, and thus we used the mean squared error loss function for those two features. Musical notes, however, are usually on certain scales, and sound better when they stay on those scales. Usually, this means striking the true, 4th, or 7th note away from the true note will sound good, while others will not. This pattern also repeats every 12 notes (since this is a 12-base scale, the -5, and -8 would be good predictions, rather than -4 and -7.) Figure 3 shows the loss function which was created by fitting a high-order polynomial, using scipy, to a set of points that satisfy these conditions. The three losses were combined by adding them together after weighing the velocity and time delta losses as 20% of the amplitude of the note-loss, which was deemed more important. This can be implemented by manually choosing appropriate loss values as a function of the difference between the predicted note and the true note

## Results
You can use the notebook in this repo to generate songs. An example in .midi format is available in the file `new_song.mid`. You can read more about our results in the paper and poster folders. 
