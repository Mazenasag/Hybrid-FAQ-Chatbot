import re
import spacy
from keybert import KeyBERT
from typing import Dict, List, Set, Tuple
import nltk
from nltk.corpus import wordnet
from collections import Counter

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class KeywordExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.kw_model = KeyBERT(model="all-mpnet-base-v2")
        self.synonym_cache = {}  # Cache for synonym lookups
        self.min_keywords = 3   # Minimum keywords to return

    def _preprocess_text(self, text: str) -> str:
        """Text normalization preserving semantic units"""
        text = re.sub(r"\b(\w+)-(\w+)\b", r"\1_\2", text)  # Preserve compounds
        text = re.sub(r"['\"](.*?)['\"]", r"\1", text)      # Remove quotes
        return text.lower()

    def _get_candidate_terms(self, doc) -> List[str]:
        """Extract candidate terms from processed document"""
        candidates = []
        for token in doc:
            # Include nouns, verbs, adjectives, and proper nouns
            if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] and not token.is_stop:
                # Handle compound nouns like "shipping method"
                if token.dep_ == "compound" and token.head.pos_ == "NOUN":
                    term = f"{token.text}_{token.head.text}"
                    candidates.append(term)
                else:
                    # Use lemma for single words
                    candidates.append(token.lemma_.lower())
        return list(set(candidates))

    def _get_contextual_synonyms(self, term: str, context_lemmas: Set[str], top_n: int = 3) -> List[str]:
        """Get synonyms that appear in the context with same POS"""
        if term in self.synonym_cache:
            return self.synonym_cache[term]
        
        # Handle multi-word terms
        base_term = term.replace('_', ' ')
        synonyms = set()
        
        # Try both noun and verb POS if needed
        pos_to_try = [wordnet.NOUN, wordnet.VERB, wordnet.ADJ] if not wordnet.synsets(base_term) else None
        
        for pos in pos_to_try or [None]:
            for synset in wordnet.synsets(base_term, pos=pos):
                for lemma in synset.lemmas():
                    syn = lemma.name().replace('_', ' ').lower()
                    # Include relevant synonyms present in context
                    if syn in context_lemmas and syn != base_term:
                        # Prefer single-word synonyms but allow multi-word
                        if ' ' in syn:
                            # Only add if all components are in context
                            if all(word in context_lemmas for word in syn.split()):
                                synonyms.add(syn)
                        else:
                            synonyms.add(syn)
                    if len(synonyms) >= top_n:
                        break
                if len(synonyms) >= top_n:
                    break
            if len(synonyms) >= top_n:
                break

        self.synonym_cache[term] = list(synonyms)
        return self.synonym_cache[term]

    def _add_fallback_keywords(self, keywords: List[Tuple[str, float]], context: str, context_lemmas: Set[str]) -> List[str]:
        """Add fallback keywords when primary extraction is insufficient"""
        # Extract additional keywords directly from answer
        doc = self.nlp(context)
        answer_keywords = Counter()
        
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                lemma = token.lemma_.lower()
                if len(lemma) > 2:
                    answer_keywords[lemma] += 1
        
        # Add most frequent answer nouns not already included
        for kw, _ in answer_keywords.most_common(5):
            if kw not in [k[0] for k in keywords] and kw in context_lemmas:
                keywords.append((kw, 0.3))  # Add with medium confidence
        
        # Sort by confidence before returning
        return sorted(keywords, key=lambda x: x[1], reverse=True)

    def extract_keywords(self, qa_pair: Dict) -> List[str]:
        """Enhanced keyword extraction with fallback mechanism"""
        question = qa_pair['question']
        answer = qa_pair['answer']
        full_text = f"{question} {answer}"
        
        # Preprocess and analyze text
        preprocessed = self._preprocess_text(full_text)
        doc = self.nlp(preprocessed)
        context_lemmas = {token.lemma_.lower() for token in doc}
        
        # Extract candidate terms from entire QA pair
        candidates = self._get_candidate_terms(doc)
        
        # Semantic ranking with KeyBERT
        keywords = self.kw_model.extract_keywords(
            full_text,
            candidates=candidates,
            keyphrase_ngram_range=(1, 2),  # Allow 1-2 word phrases
            stop_words="english",
            use_mmr=True,
            diversity=0.75,
            top_n=8
        )
        
        # Add fallback keywords if needed
        if len(keywords) < self.min_keywords:
            keywords = self._add_fallback_keywords(keywords, answer, context_lemmas)
        
        # Process results with synonym expansion
        seen = set()
        final_keywords = []
        
        for kw, score in keywords:
            base_kw = kw.replace("_", " ").strip()
            
            # Add base keyword if new
            if base_kw not in seen:
                final_keywords.append(base_kw)
                seen.add(base_kw)
            
            # Add contextual synonyms (if we need more keywords)
            if len(final_keywords) < 5:
                for syn in self._get_contextual_synonyms(base_kw, context_lemmas):
                    if syn not in seen:
                        final_keywords.append(syn)
                        seen.add(syn)
                        
            # Stop when we have enough keywords
            if len(final_keywords) >= 5:
                break
        
        # Ensure minimum keyword count
        if len(final_keywords) < self.min_keywords:
            final_keywords.extend(list(context_lemmas)[:self.min_keywords - len(final_keywords)])
        
        return list(dict.fromkeys(final_keywords))[:3]  # Preserve order, remove dups

# Test the improved extractor
extractor = KeywordExtractor()

examples = [
    {
        "question": "How do I create an account?",
        "answer": "To create an account, click on the 'Sign Up' button."
    },
    {
        "question": "Can I return promotional event items?",
        "answer": "Yes, you can return a product... applicable discounts."
    },
    {
        "question": "How can I leave a product review?",
        "answer": "To leave a review, go to the product page and click 'Write Review'."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Shipping times vary depending on the destination and the shipping method chosen. Standard shipping usually takes 3-5 business days, while express shipping can take 1-2 business days."
    },
    {
        "question": "What's your return policy?",
        "answer": "We accept returns within 30 days of purchase. Items must be unused and in original packaging."
    }
]

for i, example in enumerate(examples, 1):
    print(f"Example {i}: {extractor.extract_keywords(example)}")