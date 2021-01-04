from transformers import pipeline
import os


class MemNav:
    def __init__(self, root_dir='.'):
        self.root_dir = root_dir
        self.qa = pipeline('question-answering')
        self.sum = pipeline('summarization')
        self.context = ' '.join([open(self.root_dir + '/' + file).read()
                                 for file in sorted(os.listdir(root_dir))])

    def ask(self, question):
        return self.qa(question, self.context)['answer']

    def sum(self, file):
        print('Flag')
        print(self.root_dir + '/' + file)
        print(open(self.root_dir + '/' + file).read())
        return self.sum(open(self.root_dir + '/' + file).read(), 130, 30, False)
