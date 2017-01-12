# Machine Learning Learning Roadmap
##Beginner

- Introductions from Books: 
- Videos: 
  - [Interview with Tom Mitchell](http://videolectures.net/mlas06_mitchell_itm/)
- Machine Learning Resources for Getting Started
  - Online Video Courses
    - [Stanford Machine Learning](https://www.coursera.org/learn/machine-learning)
    - [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks)
  - Overview Papers
    - [The Discipline of Machine Learning](http://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf)
    - [A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
  - Beginner Machine Learning Books
    - [Programming Collective Intelligence](https://www.safaribooksonline.com/library/view/programming-collective-intelligence/9780596529321/)
    - [Machine Learning for Hackers](https://www.safaribooksonline.com/library/view/machine-learning-for/9781449330514/)
    - [Machine Learning, 2nd Edition](https://www.safaribooksonline.com/library/view/machine-learning-2nd/9781466583283/)
    - [Data Mining: Practical Machine Learning Tools and Techniques, Third Edition](https://www.safaribooksonline.com/library/view/data-mining-practical/9780123748560/)
    - [Practical Machine Learning](https://www.safaribooksonline.com/library/view/practical-machine-learning/9781784399689/)

##Novice
  - Get started with [Python](https://www.python.org/)
    - Syntax, data types, strings, control flow, functions, classes, exceptions, networking, asynchronous task, function decorator, annotation, context manager, multiprocessing etc…
  - Start a small project for creating a Python Web Crawler application and a RestFul Service to explore data stored
  - Install and practice Python libraries
    - [pip](https://pypi.python.org/pypi/pip)
    - [asyncio](https://docs.python.org/3/library/asyncio.html)
    - [jupyter](http://jupyter.org/)
    - [scikit-learn](http://scikit-learn.org)
      - [Quick Start Tutorial](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)
      - [User Guide](http://scikit-learn.org/stable/user_guide.html)
      - [API Reference](http://scikit-learn.org/stable/modules/classes.html)
      - [Example Gallery](http://scikit-learn.org/stable/auto_examples/index.html)
      - Papers
        - [Scikit-learn: Machine Learning in Python](http://jmlr.org/papers/v12/pedregosa11a.html)
        - [API design for machine learning software: experiences from the scikit-learn project](http://arxiv.org/abs/1309.0238)
      - Books
        - [Learning scikit-learn: Machine Learning in Python](https://www.safaribooksonline.com/library/view/learning-scikit-learn-machine/9781783281930/)
        - [Building Machine Learning Systems with Python - Second Edition](https://www.safaribooksonline.com/library/view/building-machine-learning/9781784392772/)
        - [Machine Learning with scikit learn tutorial](http://amueller.github.io/sklearn_tutorial)
    - [scikit-learn](http://scikit-learn.org) is built upon the [scipy](http://www.scipy.org/) (Scientific Python) includes:
      - [numpy](http://www.numpy.org/), base n-dimensional array package
      - [scipy](http://www.scipy.org/), fundamental library for scientific computing
      - [pandas](http://pandas.pydata.org/), data structures and analysis
      - [sympy](http://www.sympy.org/), symbolic mathematics
      - [matplotlib](http://matplotlib.org/), comprehensive 2D/3D plotting
      - [ipython](http://ipython.org/), enhanced interactive console
  - [Linear Regression example in Python](http://scipy-cookbook.readthedocs.io/items/LinearRegression.html)
  - [Linear Regression using scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
  - [Logistic Regression using scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  - [Regression analysis using Python StatsModels package](http://blog.yhat.com/posts/logistic-regression-python-rodeo.html)
  - [Using Logistic Regression in Python for Data Science](http://www.dummies.com/programming/big-data/data-science/using-logistic-regression-in-python-for-data-science/)
  - [Logistic Regression and Gradient Descent (Notebook)](http://nbviewer.jupyter.org/github/tfolkman/learningwithdata/blob/master/Logistic%20Gradient%20Descent.ipynb)
  - [Regression analysis using Python StatsModels package](http://www.turingfinance.com/regression-analysis-using-python-statsmodels-and-quandl/)
  - [k Nearest Neighbours in Python](http://scikit-learn.org/stable/modules/neighbors.html)
  - **Start a project to implement a simpler algorithm like a perceptron, k-nearest neighbour or linear regression. Write little programs to demystify methods and learn all the micro-decisions required to make it work**

##Intemediate
  - Study [scikit-learn](http://scikit-learn.org), read documentation and summarize the capabilities of [scikit-learn](http://scikit-learn.org)
  - Study one of the Machine Learning Dataset from [data.gov](https://www.data.gov/)
    - Clearly describe the problem that the dataset represents
    - Summarize the data using descriptive statistics
    - Describe the structures you observe in the data and hypothesize about the relationships in the data.
    - Spot test a handful of popular machine learning algorithms on the dataset
    - Tune well-performing algorithms and discover the algorithm and algorithm configuration that performs well on the problem

  - Design small experiments using the Datasets for studying Linear Regression, or Logistic Regression, then answer a specific question and report results
  - Try to port an open source algorithm code from one language to another
  - Study Neural Networks in Python
    - [Implementing a Neural Network from scratch in Python](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
    - [A Neural Network in 11 lines of Python](http://iamtrask.github.io/2015/07/12/basic-python-network/)
    - Get to know the Neural Networks libraries
      - Caffe
      - Theano
      - TensorFlow
      - Lasagne
      - Keras
      - mxnet
      - sklearn-theano
      - nolearn
      - DIGITS
      - Blocks
      - deepy
      - pylearn2
      - Deeplearning4j
    - [Rest in Deep Learning Libraries by Language](http://www.teglor.com/b/deep-learning-libraries-language-cm569/)

##Advanced
  - Deep Learning With Python
    - Study [Tensorflow](https://www.tensorflow.org/)
    - Study [Keras](https://keras.io/), a high-level neural networks library, which allows for easy and fast prototyping (through total modularity, minimalism, and extensibility)
  - Books
    - [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/)
  - TensorFlow knowledge points
    - Graph, Session, Variable, Fetch, Feed, TensorBoard, Playground, MNIST Practice, APIs
    - Linear Regression, Logistic Regression Modeling and Training
    - Gradients and the back propagation algorithm, Activation Functions
    - CNN, RNN and LSTM, DNN
    - Unsupervised Learning, Restricted Boltzmann Machine and Collaborative Filtering with RBM
    - Auto-encoders, Deep Belief Network, GPU programming and serving
 
