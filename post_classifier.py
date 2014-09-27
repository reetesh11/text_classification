import os, re
import json
import string
from numpy import *

class_names = ["job", "freelancer", "intern", "discussion"]
number_of_classes = len(class_names)


class WordList(object):
    """ Create a whole list of words with there frequency corresponding to the a particular document.
    """

    def __init__(self):
        self.__number_of_words = 0
        self.__bag_of_words = {}

    def __add__(self, new_bog):

        for key in new_bog.__bag_of_words:
            self.__number_of_words += new_bog.__bag_of_words[key]
            if key in self.__bag_of_words:
                self.__bag_of_words[key] += new_bog.__bag_of_words[key]
            else:
                self.__bag_of_words[key] = new_bog.__bag_of_words[key]
        return self

    def words_with_freq(self):
        """ return the whole __bag_of_words"""
        return self.__bag_of_words

    def add_word(self, word):
        """ given word is added in the dictionary __bag_of_words and
                if it is already present in __bag_of_words then the frequency of the word is increased by 1"""
        self.__number_of_words += 1
        if word in self.__bag_of_words:
            self.__bag_of_words[word] += 1
        else:
            self.__bag_of_words[word] = 1

    def length(self):
        """ return the number of different words present in current __bag_of_words"""
        return len(self.__bag_of_words)

    def word_list(self):
        """" return the list of words available in __bag_of_word"""
        return self.__bag_or_words.keys()

    def word_freq(self, word):
        """ return the frequency of the given word"""
        if word in self.__bag_of_words:
            return self.__bag_of_words[word]
        else:
            return 0


class ReadPost(object):
    """ Read the Document
    """
    _dictionary = WordList()

    def __init__(self, word_list):
        #self._post_class = None
        self._word_list = WordList()
        ReadPost._dictionary = word_list

    def read_post(self, posts, training=False):
        """ This could be used for both training and testing purpose.
        And by default, it is assumed it for testing purpose.
        So, for all training dataset you need to specify training=TRUE
        """
        #self._post_class = post_name
        text = posts.lower()
        words = (''.join(char if char not in string.punctuation else ' ' + char for char in text)).split(' ')
        temp = WordList()

        for word in words:
            self._word_list.add_word(word)
            temp.add_word(word)
            if training:
                ReadPost._dictionary.add_word(word)
        return temp.words_with_freq()

    def __add__(self, other):
        # Adding the two posts
        temp = ReadPost(ReadPost._dictionary)
        temp._word_list = self._word_list + other._word_list
        return temp

    def post_len(self):
        # Return the length of the post or number of distinguish words
        return ReadPost._dictionary.length()

    def words_and_freq(self):
        # Return the whole __bag_of_words with their frequency
        return self._word_list.words_with_freq()

    def words(self):
        # Return the words of the ReadPost object
        return self._word_list.word_list()

    def freq_of_word(self, word):
        # Return the number of times a word has appeared in the given post
        fow = self._word_list
        return fow.word_freq(word)

    def bag_of_words(self):
        #Returns the whole dictionary
        dictionary = ReadPost._dictionary.word_list()
        return dictionary


class ReadFile(ReadPost):
    """ Read the JSON file and convert it into list of words
    """
    _dictionary = WordList()
    _create_datafile = ReadPost(_dictionary)

    def __init__(self):
        self.dataset = {}
        self.post_vs_class = []
        self.word_list_per_post = {}
        self.number_of_posts = 0
        self.post_vs_words = []
        self.class_vs_words = []
        self._dictionary = WordList()

    def create_data(self, directory):
        number_of_posts = 0
        direc = os.listdir(directory)
        for files in direc:
            file_name = directory + "/" + files
            data_file = open(file_name, 'r')
            data = json.load(data_file)
            flag = 1

            for posts in data['posts']:
                """ in dataset we are using description as key and job type as value"""
                number_of_posts += 1
                self.dataset[str(posts['description'])] = str(posts['type'])
                temp = zeros(number_of_classes)
                index = class_names.index(str(posts['type']))
                temp[index] = 1
                temp = array(temp)
                self.word_list_per_post[number_of_posts] = ReadFile._create_datafile.read_post(str(posts['description'])
                                                                                               , training=True)

                if flag:
                    self.post_vs_class = temp
                    flag = 0
                else:
                    self.post_vs_class = concatenate((self.post_vs_class, temp), axis=0)

        self.post_vs_class = self.post_vs_class.reshape((number_of_posts, 4)).transpose()
        self.number_of_posts = number_of_posts
        print self.post_vs_class.shape
        #print len(self.word_list_per_post)
        #print self.dataset

    def tabulate_data(self):
        """ word_list is __bag_of_word() of the given post
        it evaluates a column of array corresponding the word_list of post
        it appends the words with their respective frequencies
        """
        bag_of_words = ReadFile._dictionary.words_with_freq()
        length_of_dict = len(bag_of_words)

        post_vs_words = zeros((length_of_dict, self.number_of_posts))
        class_vs_words = zeros((length_of_dict, number_of_classes))

        for (key, words) in self.word_list_per_post.items():
            for word, freq in words.items():
                post_vs_words[bag_of_words.keys().index(word), key-1] = freq
        self.post_vs_words = post_vs_words

        for i in range(self.number_of_posts):
            temp = post_vs_words[:, i]
            for j in range(number_of_classes):
                if post_vs_words[j, i]:
                    class_vs_words[:, j] += temp
        self.class_vs_words = class_vs_words

        print post_vs_words.shape
        print class_vs_words.shape

    def predict(self, post, jth_class=""):
        """ calculates the probability for a class jth_class given a document
        """
        _dictionary = WordList()
        bag_of_words = ReadFile._dictionary.words_with_freq()
        length_of_dict = len(bag_of_words)
        sum_words_in_all_class = sum(self.class_vs_words, axis=1)
        total_words_in_all_class = sum(self.class_vs_words, axis=0)
        number_of_documents_per_class = sum(self.post_vs_class, axis=0)

        if jth_class:
            total_num_words = total_words_in_all_class[class_names.index(jth_class)]
            prob = 0

            new_post = ReadPost(_dictionary)
            new_words = new_post.read_post(post)

            for j in range(number_of_classes):
                sum_j = total_words_in_all_class[j]
                prod = 1

                for word in new_words.keys():
                    if word in bag_of_words.keys():
                        wf = 1 + self.class_vs_words[bag_of_words.keys().index(word), j]
                        wf_jth_class = 1 + self.class_vs_words[bag_of_words.keys().index(word), class_names.index(jth_class)]
                        r = wf*total_num_words/(wf_jth_class * sum_j)
                        prod *= r

                prob += prod * self.number_of_posts/ number_of_documents_per_class[j]

            if prob != 0:
                return 1/prob
            else:
                return -1

        else:
                prob_list = []
                for jth_class in class_names:
                    prob = self.predict(post, jth_class)
                    prob_list.append([jth_class, prob])
                prob_list.sort(key=lambda x: x[1], reverse=True)
                return prob_list




directory_name = "/home/reetesh/Desktop/kroove/git/training_file"
make_data = ReadFile()
make_data.create_data(directory_name)
make_data.tabulate_data()
new_post = " hi I want to hire a person in programming"
post_name = make_data.predict(new_post)
print post_name
