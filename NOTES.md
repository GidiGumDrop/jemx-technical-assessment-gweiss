# NOTES

## Assumptions
- 184 shifts have no clock-out so the risk should be a bit higher than predicted. They were left as zero hours.
- 126 pairs of shifts have one person at two sites at the same time. They are flagged on the dashboard.
- For creating actionable feedback, I predicted what shifts will be for the week to provide actionable feedback. We do this by averaging over previous weeks. With the real roster, we can solve this optimally using Answer Set Programming and a Clingo Solver.


# Sorting Notes
- I had LLM categorise the notes (it actually categorised them based on basic lookup of words) and then built a Naive Bayes classifier.
- I had an LLM create out of sample datapoints to test, this was compared to if-then-else statements for just looking for keywords in the notes.

## What the model learned

- Trained a logistic regression model on predicting the risk score.
- Data comes from previous weeks, if we had more data, we would hold out entire sites or weeks instead of random, thus allowing us to perform proper validation on a true test set with known answers.
- A decision boundary was chosen that maximises the F1 score.
- Not confident in the results due to the data have high low signal to noise ratio so there is not much to extract.