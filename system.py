""" Main system """

import spacy
from enum import Enum

from retrieval_ranking import SemanticSearch
from retrieval_ranking import CreateLogger


class PhraseType(Enum):
    DATE = "DATE"
    CARDINAL = "CARDINAL"
    NAME = "NAME"
    GENERAL = "GENERAL"


class System(object):
    def __init__(self):
        self._logger = CreateLogger()
        self.semantic_search = SemanticSearch()

        self.nlp = spacy.load("en_core_web_sm")

    def set_ss_extractor(self, extractor, ngram_min=1, ngram_max=3):
        """ """
        if self.semantic_search:
            self.semantic_search.set_extractor(extractor, ngram_min, ngram_max)

    def set_ss_scorer(self, scorer, model_fpath="", scorer_type=""):
        """ """
        if self.semantic_search:
            self.semantic_search.set_scorer(scorer, model_fpath, scorer_type)

    def set_text(self, text, scorer):
        """ """
        for system in [self.semantic_search]:
            if system:
                system.set_text(context=text, contextual=False, scorer=scorer)
                return system.sentences

    def add_oracles(self, set_oracle):
        for system in [self.semantic_search]:
            if system:
                system.add_oracles(set_oracle)

    def search(self, query, phrases=None, top_n=10):
        # get query type
        query_type = self._get_type(query)
        self._logger.debug("query type: %s", query_type)

        # search
        result = self.semantic_search.search(query, phrases=phrases, top_n=top_n)
        result.sort(key=lambda x: x['score'], reverse=True)

        return result

    def search_with_indices(self, query, candidates=None, token_sentences=None, top_n=10, contextual=False):
        # search with indices
        result = self.semantic_search.search_for_demo(query, candidates=candidates, token_sentences=token_sentences, top_n=top_n, contextual=contextual)
        result.sort(key=lambda x: x['score'], reverse=True)

        return result

    def _get_type(self, query):
        """ """
        doc = self.nlp(query)

        for ent in doc.ents:
            if ent.label_ == "DATE":
                return PhraseType.DATE

            elif ent.label_ == "CARDINAL":
                return PhraseType.CARDINAL

            elif ent.label_ in ["ORG", "NORP", "LOC", "PERSON"]:
                return PhraseType.NAME

        return PhraseType.GENERAL


