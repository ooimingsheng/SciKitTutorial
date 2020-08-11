# import the libraries to access data 
from sklearn import datasets, neighbors

# load in the data into a variable 'digits'
digits = datasets.load_digits()

# separate the loaded data into the feature vectors x and their corresponding labels y
x_digits = digits.data
y_digits = digits.target

# find the number of instances in our dataset
n_samples = len(x_digits)
print ('Number of examples: %d' % n_samples)

# Look at the input (x) and label (y) for a particular jth instance
j = 0
print('Input #%d:' % j, x_digits[j])
print('Label #%d:' % j, y_digits[j])

# set aside the first 90% of the data for the training and the remaining 10% for testing.
x_train = x_digits[0:int(.9 * n_samples)]
y_train = y_digits[0:int(.9 * n_samples)]
x_test = x_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]
print('Number of training examples: %d' % len(y_train))
print('Number of testing examples: %d' % len(y_test))

# create an instance of the learner
knn = neighbors.KNeighborsClassifier()

# learn an h from the training data 
knn.fit(x_train, y_train)

# evaluate h over the testing data and print its accuracy.  We'll save its performance to show we did some work in class!
Q05 = knn.score(x_test,y_test)             
print('KNN score: %f' % Q05)