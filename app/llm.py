import logging
import openai
from typing import Dict, List, Tuple, Any
import re
import os

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, config):
        self.config = config
        self.api_key = config.openai_api_key
        self.model = config.llm_model
        
        # Setup OpenAI API key
        openai.api_key = self.api_key
    
    def build_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a prompt for the LLM using the question and retrieved context chunks.
        
        Args:
            question: The user's question
            context_chunks: List of retrieved text chunks relevant to the question
            
        Returns:
            Formatted prompt for the LLM
        """
        # Combine the chunks into a single context string with page citations
        context_texts = []
        for chunk in context_chunks:
            content = chunk["content"]
            page = chunk["metadata"]["page"]
            context_texts.append(f"[Page {page}]: {content}")
        
        context = "\n\n".join(context_texts)
        
        # Build the prompt
        prompt = f"""Answer the following question based ONLY on the provided context from a PDF document.
        
CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY information from the provided context.
2. If the context doesn't contain the information needed to answer the question, respond with "I cannot answer this question based on the provided document."
3. Cite the specific page numbers where you found the information (e.g., [Page X]).
4. Be concise and accurate.

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
        """
        Generate an answer to the question using the LLM and the retrieved context.
        
        Args:
            question: The user's question
            context_chunks: List of retrieved text chunks relevant to the question
            
        Returns:
            Tuple containing the answer and a list of cited page numbers
        """
        try:
            # Build the prompt
            prompt = self.build_prompt(question, context_chunks)
            
            # Call the OpenAI API
            logger.info(f"Generating answer using model: {self.model}")
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document excerpts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more factual responses
                    max_tokens=1000
                )
                
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error with OpenAI API: {e}")
                # Fallback answer generation without LLM
                logger.warning("Using fallback answer generation")
                context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])
                answer = (
                    f"I found the following information in the document that may help answer your question about '{question}':\n\n"
                    f"{context_text}\n\n"
                    f"The above excerpts are from the document and may contain relevant information to answer your question."
                )
            
            # Extract citation page numbers using regex
            citations = list(set(re.findall(r'\[Page\s+(\d+)\]', answer)))
            citations = [int(page) for page in citations]
            
            # If no citations were found through regex but we have context chunks, use their page numbers
            if not citations and context_chunks:
                citations = list(set(chunk["metadata"]["page"] for chunk in context_chunks))
            
            logger.info(f"Generated answer with {len(citations)} citations")
            return answer, citations
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
