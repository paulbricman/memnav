from transformers import pipeline
import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from nltk import sent_tokenize
from itertools import chain


class MemNav:
    def __init__(self, root_dir='.'):
        self.root_dir = root_dir
        self.qa = pipeline('question-answering')
        self.sum = pipeline('summarization')

        # Load list of entries
        self.entries = [open(self.root_dir + '/' + file).read() for file in sorted(os.listdir(root_dir))]

        # Tokenize entries in sentences
        self.entries = [sent_tokenize(entry.strip()) for entry in self.entries]

        # Merge each 3 consecutive sentences into one passage
        self.entries = list(chain(*[[' '.join(entry[start_idx:min(start_idx + 3, len(entry))]) for start_idx in range(0, len(entry), 3)] for entry in self.entries]))

        self.bi_encoder = SentenceTransformer('msmarco-distilbert-base-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')
        self.corpus_embeddings = self.bi_encoder.encode(self.entries, show_progress_bar=True)

    def retrieval(self, query):
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=100)[0]

        cross_scores = self.cross_encoder.predict([[query, self.entries[hit['corpus_id']]] for hit in hits])

        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

        results = []
        for hit in hits[:5]:
            if hit['cross-score'] > 1e-3:
                results += [self.entries[hit['corpus_id']]]
        
        return results

    def search(self, query):
        print(*self.retrieval(query), sep='\n\n')

    def ask(self, question):
        return self.qa(question, ' '.join(self.retrieval(question)))['answer']

    def summarize(self, query):
        return self.sum(' '.join(self.retrieval(query)), 130, 30, False)[0]['summary_text']
