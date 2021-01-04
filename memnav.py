from transformers import pipeline
import os


class MemNav:
    def __init__(self, root_dir='.'):
        self.qa = pipeline('question-answering')
        self.context = ' '.join([open(root_dir + '/' + file).read()
                                 for file in sorted(os.listdir(root_dir))])

    def ask(self, question):
        return self.qa(question, self.context)['answer']
