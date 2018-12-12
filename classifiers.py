from sklearn import tree
from sklearn import neural_network
from sklearn import naive_bayes

# Training data
X = [
    [181, 80, 44],
    [177, 70, 43],
    [160, 60, 38],
    [154, 54, 37],
    [166, 65, 40],
    [190, 90, 47],
    [175, 64, 39],
    [177, 70, 40],
    [159, 55, 37],
    [171, 75, 42],
    [181, 85, 43]
]

# Labels
Y = [
    'male',
    'female',
    'female',
    'female',
    'male',
    'male',
    'male',
    'female',
    'male',
    'female',
    'male'
]

# Data to be predicted
x = [[190, 70, 43]]

# Prediction model #1
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X, Y)

prediction = classifier.predict(x)
probability = classifier.predict_proba(x)

print prediction
print probability

# Prediction model #2
classifier = neural_network.MLPClassifier(alpha = 1, max_iter = 10000)
classifier = classifier.fit(X, Y)

prediction = classifier.predict(x)
probability = classifier.predict_proba(x)

print prediction
print probability

# Prediction model #3
classifier = naive_bayes.GaussianNB()
classifier = classifier.fit(X, Y)

prediction = classifier.predict(x)
probability = classifier.predict_proba(x)

print prediction
print probability