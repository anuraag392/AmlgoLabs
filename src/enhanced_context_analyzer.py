import re
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class EnhancedContextAnalyzer:
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        self.question_patterns = {
            'what': ['what is', 'what are', 'what does', 'what can', 'what will', 'what about'],
            'how': ['how does', 'how can', 'how to', 'how do', 'how is', 'how are'],
            'why': ['why is', 'why does', 'why do', 'why are', 'why would'],
            'when': ['when is', 'when does', 'when do', 'when can', 'when will'],
            'where': ['where is', 'where does', 'where can', 'where are'],
            'who': ['who is', 'who are', 'who can', 'who does'],
            'which': ['which is', 'which are', 'which can', 'which does']
        }
        
        self.intent_keywords = {
            'definition': ['what is', 'define', 'definition', 'meaning', 'means'],
            'explanation': ['explain', 'how does', 'why does', 'describe'],
            'comparison': ['difference', 'compare', 'versus', 'vs', 'better', 'worse'],
            'procedure': ['how to', 'steps', 'process', 'procedure', 'method'],
            'list': ['list', 'types', 'kinds', 'examples', 'categories'],
            'benefits': ['benefits', 'advantages', 'pros', 'good', 'positive'],
            'problems': ['problems', 'issues', 'disadvantages', 'cons', 'negative'],
            'requirements': ['requirements', 'need', 'must', 'required', 'necessary']
        }
        
        self.entity_patterns = {
            'company': ['company', 'corporation', 'business', 'organization', 'firm'],
            'product': ['product', 'service', 'tool', 'platform', 'system'],
            'person': ['user', 'customer', 'seller', 'buyer', 'member'],
            'process': ['process', 'procedure', 'method', 'workflow', 'steps'],
            'policy': ['policy', 'rule', 'regulation', 'guideline', 'term'],
            'feature': ['feature', 'function', 'capability', 'option', 'setting']
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        
        analysis = {
            'original_query': query,
            'cleaned_query': query_lower,
            'question_type': self._identify_question_type(query_lower),
            'intent': self._identify_intent(query_lower),
            'entities': self._extract_entities(query_lower),
            'key_terms': self._extract_key_terms(query_lower),
            'context_requirements': self._determine_context_requirements(query_lower),
            'complexity': self._assess_complexity(query_lower)
        }
        
        return analysis
    
    def rank_chunks_by_relevance(self, query_analysis: Dict[str, Any], 
                                context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not context_chunks:
            return []
        
        scored_chunks = []
        
        for chunk in context_chunks:
            # Calculate multiple scores
            enhanced_score = self._calculate_enhanced_relevance_score(query_analysis, chunk)
            keyword_score = self._calculate_pure_keyword_score(query_analysis, chunk)
            
            # Combine scores with adaptive weighting
            base_similarity = chunk.get('similarity_score', 0.0)
            
            # If semantic similarity is very low but keyword score is high, boost it
            if base_similarity < 0.3 and keyword_score > 0.7:
                final_score = keyword_score * 0.7 + enhanced_score * 0.3
            # If semantic similarity is high, trust it more
            elif base_similarity > 0.7:
                final_score = enhanced_score * 0.8 + keyword_score * 0.2
            # Balanced approach for medium similarities
            else:
                final_score = enhanced_score * 0.6 + keyword_score * 0.4
            
            chunk_with_score = chunk.copy()
            chunk_with_score['enhanced_score'] = enhanced_score
            chunk_with_score['keyword_score'] = keyword_score
            chunk_with_score['final_score'] = final_score
            scored_chunks.append(chunk_with_score)
        
        # Sort by final score
        scored_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity filtering to avoid too similar chunks
        diverse_chunks = self._apply_diversity_filtering(scored_chunks)
        
        return diverse_chunks
    
    def _calculate_pure_keyword_score(self, query_analysis: Dict[str, Any], chunk: Dict[str, Any]) -> float:
        """Pure keyword-based scoring for cases where semantic similarity fails"""
        chunk_text = chunk.get('metadata', {}).get('text', '').lower()
        original_query = query_analysis['original_query'].lower()
        
        # Extract all meaningful words from query
        query_words = [word for word in word_tokenize(original_query) 
                      if word.isalnum() and len(word) > 2 and word not in self.stop_words]
        
        if not query_words:
            return 0.0
        
        # Count exact matches
        exact_matches = sum(1 for word in query_words if word in chunk_text)
        
        # Count partial matches (for compound words)
        partial_matches = 0
        chunk_words = chunk_text.split()
        
        for query_word in query_words:
            if query_word not in chunk_text:  # Only check partial if no exact match
                for chunk_word in chunk_words:
                    if len(query_word) > 4 and len(chunk_word) > 4:
                        if query_word in chunk_word or chunk_word in query_word:
                            partial_matches += 0.5
                            break
        
        # Calculate coverage
        total_matches = exact_matches + partial_matches
        coverage_score = total_matches / len(query_words)
        
        # Bonus for high density of matches
        chunk_word_count = len(chunk_words)
        density_bonus = min(total_matches / max(chunk_word_count, 1) * 10, 0.3)
        
        return min(coverage_score + density_bonus, 1.0)
    
    def _apply_diversity_filtering(self, scored_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove very similar chunks to provide diverse results"""
        if len(scored_chunks) <= 3:
            return scored_chunks
        
        diverse_chunks = [scored_chunks[0]]  # Always include the top result
        
        for chunk in scored_chunks[1:]:
            chunk_text = chunk.get('metadata', {}).get('text', '').lower()
            
            # Check similarity with already selected chunks
            is_diverse = True
            for selected in diverse_chunks:
                selected_text = selected.get('metadata', {}).get('text', '').lower()
                
                # Simple overlap check
                chunk_words = set(chunk_text.split())
                selected_words = set(selected_text.split())
                
                if len(chunk_words) > 0 and len(selected_words) > 0:
                    overlap = len(chunk_words.intersection(selected_words))
                    overlap_ratio = overlap / min(len(chunk_words), len(selected_words))
                    
                    if overlap_ratio > 0.7:  # Too similar
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_chunks.append(chunk)
            
            if len(diverse_chunks) >= 5:  # Limit to top 5 diverse chunks
                break
        
        return diverse_chunks
    
    def generate_contextual_response(self, query_analysis: Dict[str, Any], 
                                   ranked_chunks: List[Dict[str, Any]]) -> str:
        if not ranked_chunks:
            return "I cannot find relevant information to answer your question in the provided documents."
        
        intent = query_analysis['intent']
        question_type = query_analysis['question_type']
        key_terms = query_analysis['key_terms']
        
        top_chunks = ranked_chunks[:3]
        context_text = " ".join([chunk.get('metadata', {}).get('text', '') for chunk in top_chunks])
        
        if intent == 'definition':
            return self._generate_definition_response(query_analysis, context_text, top_chunks)
        elif intent == 'explanation':
            return self._generate_explanation_response(query_analysis, context_text, top_chunks)
        elif intent == 'comparison':
            return self._generate_comparison_response(query_analysis, context_text, top_chunks)
        elif intent == 'procedure':
            return self._generate_procedure_response(query_analysis, context_text, top_chunks)
        elif intent == 'list':
            return self._generate_list_response(query_analysis, context_text, top_chunks)
        else:
            return self._generate_general_response(query_analysis, context_text, top_chunks)
    
    def _identify_question_type(self, query: str) -> str:
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return q_type
        return 'general'
    
    def _identify_intent(self, query: str) -> str:
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        
        for entity_type, keywords in self.entity_patterns.items():
            for keyword in keywords:
                if keyword in query:
                    entities.append(entity_type)
                    break
        
        words = word_tokenize(query)
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2]
        entities.extend(capitalized_words)
        
        return list(set(entities))
    
    def _extract_key_terms(self, query: str) -> List[str]:
        words = word_tokenize(query.lower())
        
        filtered_words = [
            self.stemmer.stem(word) for word in words 
            if word.isalnum() and word not in self.stop_words and len(word) > 2
        ]
        
        word_freq = Counter(filtered_words)
        key_terms = [word for word, freq in word_freq.most_common(10)]
        
        return key_terms
    
    def _determine_context_requirements(self, query: str) -> Dict[str, Any]:
        requirements = {
            'needs_definition': any(word in query for word in ['what is', 'define', 'meaning']),
            'needs_examples': any(word in query for word in ['example', 'examples', 'such as']),
            'needs_steps': any(word in query for word in ['how to', 'steps', 'process']),
            'needs_comparison': any(word in query for word in ['difference', 'compare', 'versus']),
            'needs_details': any(word in query for word in ['detail', 'explain', 'describe']),
            'scope': 'specific' if len(query.split()) < 5 else 'broad'
        }
        
        return requirements
    
    def _assess_complexity(self, query: str) -> str:
        word_count = len(query.split())
        question_words = sum(1 for pattern in self.question_patterns.values() 
                           for p in pattern if p in query)
        
        if word_count < 5 and question_words <= 1:
            return 'simple'
        elif word_count < 10 and question_words <= 2:
            return 'medium'
        else:
            return 'complex'
    
    def _calculate_enhanced_relevance_score(self, query_analysis: Dict[str, Any], 
                                          chunk: Dict[str, Any]) -> float:
        chunk_text = chunk.get('metadata', {}).get('text', '').lower()
        base_similarity = chunk.get('similarity_score', 0.0)
        
        key_terms = query_analysis['key_terms']
        entities = query_analysis['entities']
        intent = query_analysis['intent']
        original_query = query_analysis['original_query'].lower()
        
        # Enhanced term matching with fuzzy matching and synonyms
        term_score = self._calculate_advanced_term_score(key_terms, chunk_text, original_query)
        
        # Entity matching with variations
        entity_score = self._calculate_entity_score(entities, chunk_text)
        
        # Intent-based scoring
        intent_score = self._calculate_intent_score(intent, chunk_text)
        
        # Exact phrase matching bonus
        phrase_score = self._calculate_phrase_matching_score(original_query, chunk_text)
        
        # Question-answer pattern matching
        qa_score = self._calculate_qa_pattern_score(query_analysis, chunk_text)
        
        # Position and length scores
        position_score = 1.0 - (chunk.get('metadata', {}).get('chunk_index', 0) * 0.05)
        position_score = max(position_score, 0.3)
        
        length_score = self._calculate_optimal_length_score(chunk_text)
        
        # Weighted combination with emphasis on content relevance over similarity
        enhanced_score = (
            base_similarity * 0.2 +      # Reduced weight for base similarity
            term_score * 0.25 +          # Key terms matching
            phrase_score * 0.2 +         # Exact phrase matching
            qa_score * 0.15 +            # Question-answer patterns
            entity_score * 0.1 +         # Entity matching
            intent_score * 0.05 +        # Intent alignment
            position_score * 0.03 +      # Document position
            length_score * 0.02          # Optimal length
        )
        
        return enhanced_score
    
    def _calculate_advanced_term_score(self, key_terms: List[str], chunk_text: str, original_query: str) -> float:
        if not key_terms:
            return 0.0
        
        # Direct term matching
        direct_matches = sum(1 for term in key_terms if term in chunk_text)
        
        # Stemmed matching for variations
        chunk_words = word_tokenize(chunk_text)
        chunk_stems = [self.stemmer.stem(word) for word in chunk_words if word.isalnum()]
        
        stemmed_matches = sum(1 for term in key_terms if term in chunk_stems)
        
        # Partial matching for compound terms
        partial_matches = 0
        for term in key_terms:
            if len(term) > 4:  # Only for longer terms
                for chunk_word in chunk_words:
                    if len(chunk_word) > 4 and (term in chunk_word or chunk_word in term):
                        partial_matches += 0.5
                        break
        
        # Original query word matching (non-stemmed)
        query_words = [word.lower() for word in word_tokenize(original_query) 
                      if word.isalnum() and word.lower() not in self.stop_words]
        
        query_word_matches = sum(1 for word in query_words if word in chunk_text)
        
        total_score = (direct_matches + stemmed_matches * 0.8 + partial_matches + query_word_matches * 0.6)
        max_possible = len(key_terms) + len(query_words) * 0.6 + 2  # Rough max
        
        return min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def _calculate_entity_score(self, entities: List[str], chunk_text: str) -> float:
        if not entities:
            return 0.0
        
        matches = 0
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in chunk_text:
                matches += 1
            elif len(entity) > 3:
                # Partial matching for longer entities
                if any(entity_lower in word or word in entity_lower 
                      for word in chunk_text.split() if len(word) > 3):
                    matches += 0.5
        
        return min(matches / len(entities), 1.0)
    
    def _calculate_intent_score(self, intent: str, chunk_text: str) -> float:
        if intent not in self.intent_keywords:
            return 0.0
        
        intent_keywords = self.intent_keywords[intent]
        matches = sum(1 for keyword in intent_keywords if keyword in chunk_text)
        
        # Bonus for intent-specific patterns
        intent_patterns = {
            'definition': ['is a', 'refers to', 'means', 'defined as', 'known as'],
            'explanation': ['because', 'due to', 'reason', 'causes', 'results in'],
            'procedure': ['step', 'first', 'then', 'next', 'finally', 'process'],
            'list': ['include', 'such as', 'example', 'types', 'categories'],
            'comparison': ['different', 'unlike', 'compared to', 'versus', 'while']
        }
        
        if intent in intent_patterns:
            pattern_matches = sum(1 for pattern in intent_patterns[intent] if pattern in chunk_text)
            matches += pattern_matches * 0.5
        
        return min(matches / (len(intent_keywords) + 2), 1.0)
    
    def _calculate_phrase_matching_score(self, query: str, chunk_text: str) -> float:
        # Extract meaningful phrases from query (2-4 words)
        query_words = [word for word in word_tokenize(query.lower()) 
                      if word.isalnum() and word not in self.stop_words]
        
        if len(query_words) < 2:
            return 0.0
        
        phrase_matches = 0
        total_phrases = 0
        
        # Check 2-word phrases
        for i in range(len(query_words) - 1):
            phrase = f"{query_words[i]} {query_words[i+1]}"
            total_phrases += 1
            if phrase in chunk_text:
                phrase_matches += 1
        
        # Check 3-word phrases
        for i in range(len(query_words) - 2):
            phrase = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}"
            total_phrases += 1
            if phrase in chunk_text:
                phrase_matches += 2  # Higher weight for longer phrases
        
        return phrase_matches / max(total_phrases, 1)
    
    def _calculate_qa_pattern_score(self, query_analysis: Dict[str, Any], chunk_text: str) -> float:
        question_type = query_analysis['question_type']
        intent = query_analysis['intent']
        
        # Patterns that indicate good answers for different question types
        qa_patterns = {
            'what': ['is', 'are', 'means', 'refers to', 'defined as'],
            'how': ['by', 'through', 'using', 'via', 'method', 'way'],
            'why': ['because', 'due to', 'reason', 'since', 'as a result'],
            'when': ['during', 'after', 'before', 'while', 'time'],
            'where': ['in', 'at', 'on', 'location', 'place'],
            'who': ['person', 'people', 'user', 'customer', 'member']
        }
        
        score = 0.0
        if question_type in qa_patterns:
            patterns = qa_patterns[question_type]
            matches = sum(1 for pattern in patterns if pattern in chunk_text)
            score += matches / len(patterns)
        
        # Bonus for chunks that seem to provide direct answers
        answer_indicators = ['according to', 'states that', 'specifies', 'indicates', 'shows']
        answer_bonus = sum(0.2 for indicator in answer_indicators if indicator in chunk_text)
        
        return min(score + answer_bonus, 1.0)
    
    def _calculate_optimal_length_score(self, chunk_text: str) -> float:
        word_count = len(chunk_text.split())
        
        # Optimal range: 50-400 words (broader range)
        if 50 <= word_count <= 400:
            return 1.0
        elif word_count < 50:
            return word_count / 50.0
        else:
            # Less penalty for longer chunks
            return max(0.3, 400.0 / word_count)
    
    def _generate_definition_response(self, query_analysis: Dict[str, Any], 
                                    context_text: str, chunks: List[Dict[str, Any]]) -> str:
        key_terms = query_analysis['key_terms']
        
        sentences = sent_tokenize(context_text)
        definition_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in key_terms):
                if any(word in sentence_lower for word in ['is', 'are', 'means', 'refers to', 'defined as']):
                    definition_sentences.append(sentence.strip())
        
        if definition_sentences:
            response = f"Based on the provided context:\n\n{definition_sentences[0]}"
            if len(definition_sentences) > 1:
                response += f"\n\nAdditionally: {definition_sentences[1]}"
            return response
        
        return f"Based on the context, here's what I found about your query:\n\n{sentences[0] if sentences else context_text[:300]}..."
    
    def _generate_explanation_response(self, query_analysis: Dict[str, Any], 
                                     context_text: str, chunks: List[Dict[str, Any]]) -> str:
        sentences = sent_tokenize(context_text)
        
        explanation_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['because', 'due to', 'reason', 'purpose', 'in order to']):
                explanation_sentences.append(sentence.strip())
        
        if explanation_sentences:
            return f"Here's the explanation based on the context:\n\n{' '.join(explanation_sentences[:2])}"
        
        return f"Based on the provided information:\n\n{' '.join(sentences[:2])}"
    
    def _generate_comparison_response(self, query_analysis: Dict[str, Any], 
                                    context_text: str, chunks: List[Dict[str, Any]]) -> str:
        sentences = sent_tokenize(context_text)
        
        comparison_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['different', 'unlike', 'compared to', 'versus', 'while']):
                comparison_sentences.append(sentence.strip())
        
        if comparison_sentences:
            return f"Based on the context, here are the key differences:\n\n{' '.join(comparison_sentences)}"
        
        return f"Here's what the context provides for comparison:\n\n{' '.join(sentences[:2])}"
    
    def _generate_procedure_response(self, query_analysis: Dict[str, Any], 
                                   context_text: str, chunks: List[Dict[str, Any]]) -> str:
        sentences = sent_tokenize(context_text)
        
        step_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['first', 'second', 'then', 'next', 'finally', 'step']):
                step_sentences.append(sentence.strip())
        
        if step_sentences:
            response = "Based on the context, here are the steps:\n\n"
            for i, step in enumerate(step_sentences[:5], 1):
                response += f"{i}. {step}\n"
            return response
        
        return f"Here's the process information from the context:\n\n{' '.join(sentences[:3])}"
    
    def _generate_list_response(self, query_analysis: Dict[str, Any], 
                              context_text: str, chunks: List[Dict[str, Any]]) -> str:
        sentences = sent_tokenize(context_text)
        
        list_items = []
        for sentence in sentences:
            if any(char in sentence for char in ['•', '-', '*']) or sentence.strip().startswith(tuple('123456789')):
                list_items.append(sentence.strip())
        
        if list_items:
            response = "Based on the context, here are the items:\n\n"
            for item in list_items[:7]:
                response += f"• {item}\n"
            return response
        
        bullet_points = re.findall(r'[•\-\*]\s*([^•\-\*\n]+)', context_text)
        if bullet_points:
            response = "Here are the key points from the context:\n\n"
            for point in bullet_points[:7]:
                response += f"• {point.strip()}\n"
            return response
        
        return f"Based on the context:\n\n{' '.join(sentences[:3])}"
    
    def _generate_general_response(self, query_analysis: Dict[str, Any], 
                                 context_text: str, chunks: List[Dict[str, Any]]) -> str:
        key_terms = query_analysis['key_terms']
        sentences = sent_tokenize(context_text)
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in key_terms):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return f"Based on the provided context:\n\n{' '.join(relevant_sentences[:2])}"
        
        return f"Here's what I found in the context related to your question:\n\n{' '.join(sentences[:2])}"
           