import re
import glob
import decimal
import random
import math


class Bayes_Structure:
    def __init__(self):
        self.classes = dict()
        self.vocabulary = set()
        self.stopwords = self.get_stopwords()

    def get_stopwords(self):
        stopfile = open('stopwords.txt')
        stopwords = stopfile.read()
        stopwords = stopwords.replace('\n', ' ')
        stopwords = stopwords.replace('\t', ' ')
        stopwords = stopwords.split(' ')
        stopset = set()
        stopset.update(stopwords)
        return stopwords

    def train(self, file, file_class):
        stripped = self.strip_file(file)
        to_train = self.classes.get(file_class, Bayes_Class())
        to_train.new_training_document(stripped)
        self.classes[file_class] = to_train
        self.vocabulary.update(stripped)

    def strip_file(self, file_to_strip):
        file = open(file_to_strip, 'r', encoding='iso-8859-1')
        content = file.read()
        segments = content.split('\n\n')
        without_header = ' '.join(segments[1:])
        lines = without_header.split('\n')
        valid_lines = list()
        for line in lines:
            found = re.search('[a-z0-9\\-\\.]+\\.(com|org|net|mil|edu|(co\\.[a-z].))', line)
            if found is None:
                valid_lines.append(line)
        valid_lines = ' '.join(valid_lines)
        stripped = re.sub(r'\W+\s\t\n', '', valid_lines)
        stripped = re.sub(r'\d+', '', stripped)
        stripped = re.sub('[^a-zA-Z]+', ' ', stripped)

        stripped = ' '.join(stripped.split())
        stripped = stripped.lower()
        words = stripped.split(' ')
        to_return = list()
        for word in words:
            if word not in self.stopwords:
                to_return.append(word)

        return to_return

    def train_class(self, file_list ,class_to_train):
        folder_path = 'Classes/' + class_to_train + '/*'
        for file in file_list:
            self.train(file, class_to_train)

    def test(self, doc_to_test):
        tuple_list = list()
        for class_holder in self.classes.keys():
            class_val = self.classes.get(class_holder)
            class_prob = class_val.doc_probability(doc_to_test, len(self.vocabulary))
            new_tup = (class_holder, class_prob)
            tuple_list.append(new_tup)
        most_likely = max(tuple_list, key=lambda item:item[1])
        return most_likely[0]

class BayesTesterTrainer:
    def __init__(self):
        self.Bayes_Structure = Bayes_Structure()
        self.testing_set = dict()

    def train(self):
        classes = glob.glob('Classes/*')
        for entry in classes:
            training_set, testing_set = self.scramble_class(entry)
            class_name = entry[8:]
            self.Bayes_Structure.train_class(training_set, class_name)
            for file in testing_set:
                self.testing_set[file] = class_name


    def scramble_class(self, entry):
        files = glob.glob(entry + '/*')
        files = sorted(files, key = lambda x: random.random())
        cut_off = math.ceil(.8 * len(files))
        training_set = files[:cut_off]
        testing_set = files[cut_off:]
        return training_set, testing_set

    def get_struct(self):
        return self.Bayes_Structure


    def test(self):
        numerator = 0
        denominator = 0
        for file in self.testing_set.keys():
            response = self.Bayes_Structure.test(file)
            if response == self.testing_set.get(file):
                numerator += 1
            denominator += 1
        accuracy = numerator / denominator
        return accuracy
class Bayes_Class:
    def __init__(self):
        self.word_counts = dict()
        self.total_words = 0

    def new_training_document(self, doc_words):
        for word in doc_words:
            self.word_counts[word] = self.word_counts.get(word, 0) + 1
            self.total_words += 1

    def doc_probability(self, doc_to_test, vocabulary_size):
        class_prob = decimal.Decimal(1)
        decimal.getcontext().prec = 800
        for word in doc_to_test:
            tf = decimal.Decimal(self.word_counts.get(word, 0) + 1)
            denom = decimal.Decimal(self.total_words + vocabulary_size)
            word_prob = (tf / denom)
            class_prob *= word_prob
        return class_prob


#Classes/comp.graphics/37261
'''
temp = Bayes_Structure()
temp.train_class('comp.graphics')
doc = temp.strip_file('Classes/comp.graphics/37261')
verdict = temp.test(doc)'''
temp = BayesTesterTrainer()
temp.train()
temp.test()
print('here')

