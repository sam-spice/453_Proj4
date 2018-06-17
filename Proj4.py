import re


class Bayes_Structure:
    def __init__(self):
        self.classes = dict()
        self.vocabulary = set()



class Bayes_Class:
    def __init__(self):
        self.word_counts = dict()
        self.total_words = 0

    def new_training_document(self, doc_words):
        for word in doc_words:
            self.word_counts[word] = self.word_counts.get(word, 0) + 1
            self.total_words += 1

def get_stopwords():
    stopfile = open('stopwords.txt')
    stopwords = stopfile.read()
    stopwords = stopwords.replace('\n', ' ')
    stopwords = stopwords.replace('\t', ' ')
    stopwords = stopwords.split(' ')
    stopset = set()
    stopset.update(stopwords)
    return stopwords

def strip_file(file_to_strip, stopset):
    file = open(file_to_strip, 'r')
    content = file.read()
    segments = content.split('\n\n')
    without_header = ' '.join(segments[1:])
    stripped = re.sub(r'\W+\s\t\n', '', without_header)
    stripped = re.sub(r'\d+', '', without_header)
    stripped = re.sub('[^a-zA-Z]+', ' ', stripped)



    stripped = ' '.join(stripped.split())
    stripped = stripped.lower()
    words = stripped.split(' ')
    to_return = list()
    for word in words:
        if word not in stopset:
            to_return.append(word)


    return to_return


stopset = get_stopwords()
strip_file('Classes/comp.graphics/37261', stopset)