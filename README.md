# Predict-Yelp-Review-
Yelp reviews are used to train a classifier using keras libraries to predict if a review is positive (4 or 5) or negative (1, 2 or 3)

When the ANN is used, I got an accuracy of 82% on the test set. No tuning was done. 
confusion_matrix(y_test, predictions_bow > 0.5)
Out[613]: 
array([[ 521,  296],
       [ 158, 1525]])
       

When a Multinomial NB is done, I got a 58% accuracy. 

Also, I got a better accuracy when I used a bag of words input rather than the ttfidf transformation. 
