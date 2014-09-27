import os, re
import json
import sys
import math
import numpy
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC as SVM
from collections import defaultdict


class ReadFile(object):
    """ Read the JSON file and convert it into list of words
    """

    def __init__(self):
        self.dataset = {}
        self.__name = ""
        self.__post_class = None

    def create_dataset(self, file_name):
        """ read the given file and create a list of words """
        data_file = open(file_name, 'r')
        data = json.load(data_file)

        for posts in data['posts']:
            """ in temp we are using description as key and job type as value"""
            self.dataset[str(posts['description'])] = str(posts['type'])

        return self.dataset

    def posts_and_target(self, dataset):
        input_data = []
        output_data = []
        for posts in dataset.keys():
            input_data.append(posts)
            output_data.append(dataset[posts])

        return input_data, output_data

    def read_posts(self, post):
        # reads the given post and append the words in the post to the bag_of_words
        post = post.lower()
        words = re.split("[^\w]*", post)

        for word in words:
            self.__word_list.add_word(word)

        return self.__word_list


class ClassifierWrapper(object):

    def __init__(self):
        self.vectorizer = DictVectorizer()
        self.classifier = SVM()
        self.tfidf_transformer = TfidfTransformer()
        self.count_vec = CountVectorizer()
        self.clf = MultinomialNB()

    def train(self, dataset):

        """ Train the dataset with above specified classifier """
        _X_train = self.count_vec.fit_transform(dataset)
        tf_transformer = TfidfTransformer(use_idf=False).fit(_X_train)
        _X_train_tf = tf_transformer.transform(_X_train)
        _X_train_tfidf = self.tfidf_transformer.fit_transform(_X_train)
        posts, target = ReadFile().posts_and_target(dataset)
        self.clf.fit(_X_train_tfidf, target)

    def predict(self, in_bags):
        _X_new_count = self.count_vec.transform(in_bags)
        _X_new_tfidf = self.tfidf_transformer.transform(_X_new_count)
        predicted = self.clf.predict(_X_new_tfidf)

        return predicted

new_file = ReadFile()
classify = ClassifierWrapper()
data = new_file.create_dataset("/home/reetesh/Desktop/kroove/dataset.json")

classify.train(data)
new_post = " hi I want to hire a person who has a good background in programming"
print classify.predict(new_post)[0]

