# dslr---Datascience-X-Logistic-Regression
Some Datascience bases and a logistic regression implementation
# Data Science: Logistic Regression

## Introduction

On no! Since its creation, the famous school of wizards, Hogwarts, had never known such
an offense. The forces of evil have bewitched the Sorting Hat. It no longer responds, and
is unable to fulfill his role of sorting the students to the houses.

The new academic year is approaching. Gladly, the Professor McGonagall was able
to take action in such a stressful situation, since it is impossible for Hogwarts not to
welcome new students... She decided to call on you, a muggle "datascientist" who is
able to create miracles with the tool which all muggles know how to use: a "computer".

Despite the intrinsic reluctance of many wizards, the director of the school welcomes
you to his office to explain the situation. You are here because his informant discovered
that you are able to recreate a magic Sorting Hat using your muggle tools. You explain
to him that in order for your "muggle" tools to work, you need students data. Hesi-
tantly, Professor McGonagall gives you a dusty spellbook. Fortunately for you, a simple
"Digitalis!" and the book turned into a USB stick.


## Chapter I

## Objectives

In this project _DataScience x Logistic Regression_ , you will continue your exploration of
_Machine Learning_ by discovering different tools.

The use of the term _DataScience_ in the title will be clearly considered by some to be
abusive. That is true. We do not pretend to give you all the basics of _DataScience_ in this
topic. The subject is vast. We will only see here some bases which seemed to us useful
for data exploration before sending it to the _machine learning_ algorithm.

You will implement a linear classification model, as a continuation of the subject _lin-
ear regression_ : a _logistic regression_. We also encourage you a lot to create a _machine
learning_ toolkit while you will move along the branch.

```
Summarizing:
```
- You will learn how to read a data set, to visualize it in different ways, to select and
    clean unnecessary information from your data.
- You will train a logistic regression that will solve classification problem.


# Chapter II

# General instructions

You can use whatever language you want. However, we recommend that you choose a
language with a library that facilitates plotting and calculation of statistical properties
of a dataset.

Any functions that would do all the heavy-lifting for you (for example, using the
describefunction of the _Pandas_ ) library will be considered cheating.


# Chapter III

# Mandatory part

```
It is highly recommended to perform the steps in following order.
```
### III.1 Data Analysis

```
We will see some basic steps of data exploration. Of course, these
are not the only techniques available or the only one step to follow.
Each data set and problem has to be approached in an unique way. You
will surely find other ways to analyze your data in the future.
```
First of all, take a look at the available data. look in what format it is presented, if
there are various types of data, the different ranges, and so on. It is important to make
an idea of your raw material before starting. The more you work on data - the more you
develop an intuition about how you will be able to use it.

In this part, Professor McGonagall asks you to produce a program calleddescribe.[extension].
This program will take a _dataset_ as a parameter. All it has to do is to display information
for all numerical _features_ like in the example:

```
$> describe.[extension] dataset_train.csv
Feature 1 Feature 2 Feature 3 Feature 4
Count 149.000000 149.000000 149.000000 149.
Mean 5.848322 3.051007 3.774497 1.
Std 5.906338 3.081445 4.162021 1.
Min 4.300000 2.000000 1.000000 0.
25% 5.100000 2.800000 1.600000 0.
50% 5.800000 3.000000 4.400000 1.
75% 6.400000 3.300000 5.100000 1.
Max 7.900000 4.400000 6.900000 2.
```

Datascience X Logistic Regression Harry Potter and a Data Scientist

```
It is forbidden to use any function that makes the job done for you
like: count, mean, std, min, max, percentile, etc... no matter the
language that you use. Of course, it is also forbidden to use the
describe library or any function that looks similar(more or less) to
it from another library.
```
### III.2 Data Visualization

Data visualization is a powerful tool for a data scientist. It allows you to make insights
and develop an intuition of what your data looks like. Visualizing your data also allows
you to detect defects or anomalies.

In this section, you are asked to create a set of scripts, each using a particular visu-
alization method to answer a question. There is not necessarily a single answer to the
question.

#### III.2.1 Histogram

Make a script calledhistogram.[extension]which displays a _histogram_ answering the
next question :

```
Which Hogwarts course has a homogeneous score distribution between all four houses?
```
#### VIII.2.2 Scatter plot

Make a script calledscatter_plot.[extension]which displays a _scatter plot_ answering
the next question :

```
What are the two features that are similar?
```
#### III.2.3 Pair plot

Make a script calledpair_plot.[extension]which displays a _pair plot_ or _scatter plot
matrix_ (according to the library that you are using).

```
From this visualization, what features are you going to use for your logistic regression?
```

Datascience X Logistic Regression Harry Potter and a Data Scientist

### III.3 Logistic Regression

You arrive at the last part: code your Magic Hat. To do this, you have to perform a
multi-classifier using a logistic regression _one-vs-all_.

```
You will have to make two programs :
```
- First one will _train_ your models, it’s calledlogreg_train.[extension]. It takes
    as a parameterdataset_train.csv.. For the mandatory part, you must use the
    technique of _gradient descent_ to minimize the error. The program generates a file
    containing the weights that will be used for the prediction.
- A second has to be namedlogreg_predict.[extension]. It takes as a parameter
    dataset_test.csvand a file containing the weights trained by previous program.

```
In order to evaluate the performance of your classifier this second program will have
to generate a prediction filehouses.csvformatted exactly as follows:
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Hufflepuff
4,Slytherin
5,Ravenclaw
6,Hufflepuff
[...]
```
