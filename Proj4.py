import re
import glob
import decimal
import random
import math
import sys
import datetime


class Bayes_Structure:
    def __init__(self):
        self.classes = dict()
        self.vocabulary = set()
        self.stopwords = self.get_stopwords()
        self.class_count = dict()
        self.total_docs = 0
        self.word_in_doc_counts = dict()
        self.feature_list = None

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
        words_in_doc = to_train.new_training_document(stripped)
        for word in words_in_doc:
            self.word_in_doc_counts[word] = self.word_in_doc_counts.get(word, 0) + 1
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
        self.total_docs += len(file_list)
        for file in file_list:
            self.train(file, class_to_train)
            self.class_count[class_to_train] = self.class_count.get(class_to_train, 0) + 1

    def information_gain(self, word):
        segment1 = 0
        segment2 = 0
        segment3 = 0
        for class_ in self.classes.keys():
            class_struct = self.classes.get(class_)
            total_class_docs = class_struct.num_docs
            segment1 += (total_class_docs / self.total_docs) * math.log2((total_class_docs / self.total_docs))

            p_c_giv_w = (class_struct.word_in_docs.get(word, 0) / self.word_in_doc_counts.get(word))
            if p_c_giv_w != 0:
                segment2 += p_c_giv_w * math.log2(p_c_giv_w)

            p_c_without_w = (total_class_docs - class_struct.word_in_docs.get(word, 0)) / \
                            (self.total_docs - self.word_in_doc_counts.get(word))
            if p_c_without_w != 0:
                segment3 += p_c_without_w * math.log2(p_c_without_w)

        p_w = self.word_in_doc_counts.get(word) / self.total_docs
        p_not_w = (self.total_docs - self.word_in_doc_counts.get(word)) / self.total_docs
        ig = - segment1 + (p_w * segment2) + (p_not_w * segment3)
        return ig

    def feature_selection(self, m):
        feature_list = list()
        for word in self.vocabulary:
            ig = self.information_gain(word)
            new_tup = (word, ig)
            feature_list.append(new_tup)
        sorted_features = sorted(feature_list, key=lambda item:item[1], reverse=True)
        feature_words = list()
        if m < len(self.vocabulary):
            for i in range(m):
                feature_words.append(sorted_features[i][0])
        else:
            for i in range(len(self.vocabulary)):
                feature_words.append(sorted_features[i][0])
        self.feature_list = feature_words

    def test(self, doc_to_test):
        tuple_list = list()
        doc_words = self.strip_file(doc_to_test)
        for class_holder in self.classes.keys():
            class_val = self.classes.get(class_holder)
            class_prob = class_val.doc_probability(doc_words, len(self.vocabulary), self.feature_list)
            new_tup = (class_holder, class_prob)
            tuple_list.append(new_tup)
        most_likely = max(tuple_list, key=lambda item:item[1])
        return most_likely[0]

class BayesTesterTrainer:
    def __init__(self):
        self.Bayes_Structure = Bayes_Structure()
        self.testing_set = dict()

    def train(self, feature_number):
        classes = glob.glob('Classes/*')
        for entry in classes:
            training_set, testing_set = self.scramble_class(entry)
            class_name = entry[8:]
            self.Bayes_Structure.train_class(training_set, class_name)
            for file in testing_set:
                self.testing_set[file] = class_name
        if 0 < feature_number < len(self.Bayes_Structure.vocabulary):
            self.Bayes_Structure.feature_selection(feature_number)


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
                # print(response)
                numerator += 1
            denominator += 1
        accuracy = numerator / denominator
        return accuracy

class Bayes_Class:
    def __init__(self):
        self.word_counts = dict()
        self.total_words = 0
        self.num_docs = 0
        self.word_in_docs = dict()

    def new_training_document(self, doc_words):
        self.num_docs += 1
        unique_words = set()
        for word in doc_words:
            unique_words.add(word)
            self.word_counts[word] = self.word_counts.get(word, 0) + 1
            self.total_words += 1
        for unique in unique_words:
            self.word_in_docs[unique] = self.word_in_docs.get(unique, 0) + 1
        return unique_words

    def doc_probability(self, doc_to_test, vocabulary_size, feature_list):
        class_prob = decimal.Decimal(1)
        decimal.getcontext().prec = 800
        for word in doc_to_test:
            if feature_list is not None:
                if word not in feature_list:
                    continue
            tf = decimal.Decimal(self.word_counts.get(word, 0) + 1)
            denom = decimal.Decimal(self.total_words + vocabulary_size)
            word_prob = (tf / denom)
            class_prob *= word_prob
        return class_prob

def main():
    feature_size = int(sys.argv[1])
    trainer_tester = BayesTesterTrainer()
    print('Feature Size: ' + str(feature_size))

    training_start = datetime.datetime.now()
    trainer_tester.train(feature_size)
    training_end = datetime.datetime.now()

    training_delta = training_end - training_start
    print('Training Time: ' + str(training_delta.seconds) + ' seconds')

    testing_start = datetime.datetime.now()
    accuracy = trainer_tester.test()
    testing_end = datetime.datetime.now()

    testing_delta = testing_end - testing_start
    print('Testing Time: ' + str(testing_delta.seconds) + ' seconds')
    print('Accuracy: ' + str(accuracy))



#Classes/comp.graphics/37261
'''
temp = Bayes_Structure()
temp.train_class('comp.graphics')
doc = temp.strip_file('Classes/comp.graphics/37261')
verdict = temp.test(doc)'''

main()
