from transformers import pipeline
import os


class MemNav:
    def __init__(self, root_dir='.'):
        self.root_dir = root_dir
        self.qa = pipeline('question-answering')
        self.summarize = pipeline('summarization')
        self.context = ' '.join([open(self.root_dir + '/' + file).read()
                                 for file in sorted(os.listdir(root_dir))])

    def ask(self, question):
        return self.qa(question, self.context)['answer']

    def sum(self, file):
        return self.summarize(open(self.root_dir + '/' + file).read()[:1024], 130, 30, False)['summary_text']
