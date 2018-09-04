# Machine Learning Algorithms Second Edition

<a href="https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-algorithms-second-edition?utm_source=github&utm_medium=reposiory"><img src="https://d255esdrn735hr.cloudfront.net/sites/default/files/imagecache/ppv4_main_book_cover/B10943_MockupCoverNew.png" alt="Book Name" height="256px" align="right"></a>

This is the code repository for [Machine Learning Algorithms Second Edition](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-algorithms-second-edition?utm_source=github&utm_medium=reposiory), published by Packt.

**Popular algorithms for data science and machine learning**

## What is this book about?
Machine learning has gained tremendous popularity for its powerful and fast predictions with large datasets. However, the true forces behind its powerful output are the complex algorithms involving substantial statistical analysis that churn large datasets and generate substantial insight.

This book covers the following exciting features: 
* Study feature selection and the feature engineering process
* Assess performance and error trade-offs for linear regression
* Build a data model and understand how it works by using different types of algorithm
* Learn to tune the parameters of Support Vector Machines (SVM)
* Explore the concept of natural language processing (NLP) and recommendation systems

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1789347998) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
svc = SVC(kernel='linear')
print(cross_val_score(svc, X, Y, scoring='accuracy', cv=10).mean())
0.93191356542617032
```

**Following is what you need for this book:**
Machine Learning Algorithms is for you if you are a machine learning engineer, data engineer, or junior data scientist who wants to advance in the field of predictive analytics and machine learning. Familiarity with R and Python will be an added advantage for getting the best from this book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-15).

### Software and Hardware List

| Chapter  | Software required                   | OS required                        |
| -------- | ------------------------------------| -----------------------------------|
| 2-17     | Python 2.7/3.5, SciPy 0.18,         | Windows, Mac OS X, and Linux (Any) |
|          | Numpy 1.11+, Matplotlib 2.0,        |                                    |
|          | ScikitLearn 0.18+, Crab,            |                                    |
|          | Apache Spark 2+, NLTK –langdetect,  |                                    |
|          | Gensim, Keras 2+, Cupy              |                                    |
                             


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/MachineLearningAlgorithmsSecondEdition_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Mastering Machine Learning Algorithms [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/mastering-machine-learning-algorithms?utm_source=github&utm_medium=repository&utm_campaign=9781788621113) [[Amazon]](https://www.amazon.com/dp/1788621115)

* Python Deep Learning [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/python-deep-learning?utm_source=github&utm_medium=repository&utm_campaign=9781786464453) [[Amazon]](https://www.amazon.com/dp/1786464454)

## Get to Know the Author
**Giuseppe Bonaccorso**
is an experienced team leader/manager in AI, machine/deep learning solution design, management, and delivery. He got his MScEng in electronics in 2005 from the University of Catania, Italy, and continued his studies at the University of Rome Tor Vergata and the University of Essex, UK. His main interests include machine/deep learning, reinforcement learning, big data, bio-inspired adaptive systems, cryptocurrencies, and NLP.



## Other books by the authors
* [Mastering Machine Learning Algorithms](https://www.packtpub.com/big-data-and-business-intelligence/mastering-machine-learning-algorithms?utm_source=github&utm_medium=repository&utm_campaign=9781788621113)
* [Machine Learning Algorithms](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-algorithms?utm_source=github&utm_medium=repository&utm_campaign=9781785889622)

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
