from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import numpy as np

VECTOR_DIR = "vector_store"

class AdvancedRAG:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = OllamaLLM(model="llama3.2", temperature=0.2)

    def load_vectorstore(self):
        """Load the FAISS vector store"""
        try:
            return FAISS.load_local(
                VECTOR_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise Exception(f"Could not load vector store: {str(e)}")

    def create_advanced_retriever(self, vectorstore, strategy="hybrid", k=4):
        """
        Create advanced retrievers with different strategies:
        - similarity: Standard similarity search
        - mmr: Maximal Marginal Relevance (diversity + relevance)
        - hybrid: Enhanced similarity with query expansion
        """
        if strategy == "similarity":
            return vectorstore.as_retriever(search_kwargs={"k": k})

        elif strategy == "mmr":
            return vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "lambda_mult": 0.7}
            )

        elif strategy == "hybrid":
            # For hybrid, we'll use MMR with enhanced processing
            return vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "lambda_mult": 0.7}
            )

        else:
            return vectorstore.as_retriever(search_kwargs={"k": k})

    def expand_query(self, query, job_description):
        """Expand the query with job-specific context"""
        expansion_prompt = f"""
        Based on this job description, generate 3-5 additional search queries that would help find relevant resume content:

        Job Description:
        {job_description}

        Original Query: {query}

        Generate additional queries that focus on:
        1. Specific technical skills mentioned
        2. Experience requirements
        3. Key responsibilities
        4. Preferred qualifications

        Return only the additional queries, one per line.
        """

        try:
            expansion_response = self.llm.invoke(expansion_prompt)
            additional_queries = [q.strip() for q in expansion_response.split('\n') if q.strip()]
            return [query] + additional_queries[:5]  # Limit to 5 additional queries
        except:
            return [query]

    def retrieve_with_scores(self, retriever, queries):
        """Retrieve documents with relevance scores"""
        all_docs = []
        all_scores = []

        for query in queries:
            try:
                docs = retriever.invoke(query)
                # For each doc, we can't easily get scores from LangChain retrievers
                # So we'll assign equal weight to all retrieved docs
                scores = [1.0] * len(docs)  # Placeholder scores
                all_docs.extend(docs)
                all_scores.extend(scores)
            except Exception as e:
                print(f"Error retrieving for query '{query}': {e}")
                continue

        # Remove duplicates and sort by score
        seen_content = set()
        unique_docs = []
        unique_scores = []

        for doc, score in zip(all_docs, all_scores):
            content = doc.page_content.strip()
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
                unique_scores.append(score)

        # Sort by score (descending)
        sorted_indices = np.argsort(unique_scores)[::-1]
        sorted_docs = [unique_docs[i] for i in sorted_indices[:10]]  # Top 10

        return sorted_docs

    def evaluate_single_resume(self, filename, strategy="hybrid", k=4, use_query_expansion=True):
        """Evaluate a specific resume using RAG"""
        print(f"🤖 Evaluating single resume: {filename} (Strategy: {strategy}, K: {k})...")

        vectorstore = self.load_vectorstore()
        
        # Filter documents by source filename
        all_docs = vectorstore.docstore._dict.values()
        filtered_docs = [doc for doc in all_docs if doc.metadata.get('source') == filename]
        
        if not filtered_docs:
            return f"❌ No content found for resume: {filename}"

        # Create a temporary vectorstore from filtered docs
        temp_vectorstore = FAISS.from_documents(filtered_docs, self.embeddings)
        retriever = self.create_advanced_retriever(temp_vectorstore, strategy, k)

        with open("job_description.txt", "r", encoding="utf-8") as f:
            job_description = f.read()

        base_query = """
        You are a technical recruiter evaluating a candidate.

        Tasks:
        1. Summarize the candidate's background and experience
        2. Match their skills with the job requirements
        3. Rate the candidate from 0 to 10
        4. Explain the rating clearly with specific examples
        """

        # Expand query if requested
        if use_query_expansion:
            queries = self.expand_query(base_query, job_description)
            print(f"🔍 Using {len(queries)} expanded queries")
        else:
            queries = [base_query]

        # Retrieve relevant documents
        docs = self.retrieve_with_scores(retriever, queries)

        if not docs:
            return "❌ No relevant resume content found for evaluation."

        # Prepare context
        context_parts = []
        for i, doc in enumerate(docs[:k], 1):  # Limit to k documents
            context_parts.append(f"--- Resume Section {i} ---\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Enhanced prompt with RAG context
        enhanced_query = f"""
{base_query}

Job Description:
{job_description}

Retrieved Resume Content ({len(docs)} relevant sections found):
{context}

Please provide a comprehensive evaluation based on the retrieved resume content above.
"""

        print(f"📄 Retrieved {len(docs)} relevant sections, using top {min(k, len(docs))}")

        response = self.llm.invoke(enhanced_query)

        return response

# Backward compatibility
def evaluate_candidate():
    """Legacy function for backward compatibility"""
    rag = AdvancedRAG()
    return rag.evaluate_with_rag()
