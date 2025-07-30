import os
import re
import fitz 
import tempfile
import pandas as pd
from typing import List
from datetime import datetime
import logging
from langchain_openai import OpenAIEmbeddings  
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from django.conf import settings

logger = logging.getLogger(__name__)

class AIDocumentChat:
    def __init__(self):
        """Initialize the document chat system with models and vector store."""
        # Initialize LLM based on settings
        if settings.ACTIVE_MODEL == "GROQ":
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model="llama3-70b-8192",
                api_key=settings.GROQ_EMS_QUERY_KEY
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-4o",
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.3
            )
        
        # Initialize OpenAI embeddings 
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize vector store
        self.vector_store = None
        self.uploaded_documents = []
        
        # Document processing directory
        self.documents_dir = os.path.join(settings.MEDIA_ROOT, "documents")
        os.makedirs(self.documents_dir, exist_ok=True)

    def _embed_documents(self, texts: List[str]):
        """Embed documents with batching for large documents"""
        batch_size = 32  # Adjust based on GPU memory
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_tensor=False,
                device=self.device,
                normalize_embeddings=True
            )
            embeddings.extend(batch_embeddings)
        return embeddings

    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary storage and return its path."""
        try:
            # Create a temporary file with proper extension
            file_ext = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=self.documents_dir,
                suffix=file_ext
            ) as tmp_file:
                for chunk in uploaded_file.chunks():
                    tmp_file.write(chunk)
                return tmp_file.name
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise Exception(f"Error saving file: {str(e)}")

    def _excel_to_text(self, file_path: str) -> str:
        """Convert Excel file to text format."""
        try:
            # Read Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            # Process each sheet
            text_content = []
            for sheet_name, df in excel_data.items():
                text_content.append(f"--- Sheet: {sheet_name} ---")
                
                # Add column names
                text_content.append("Columns: " + ", ".join(df.columns))
                
                # Add sample data (first 10 rows)
                text_content.append("Sample data:")
                text_content.append(df.head(10).to_string(index=False))
                
                # Add summary statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if not numeric_cols.empty:
                    text_content.append("Summary statistics:")
                    text_content.append(df[numeric_cols].describe().to_string())
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            raise Exception(f"Could not process Excel file: {str(e)}")

    def _process_document(self, file_path: str) -> List[str]:
        """Process a document (PDF, text, or Excel) and return chunks of text."""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Use appropriate loader based on file extension
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_path.lower().endswith(('.txt', '.text')):
                loader = TextLoader(file_path)
                documents = loader.load()
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                # Process Excel file
                excel_text = self._excel_to_text(file_path)
                from langchain_core.documents import Document
                documents = [Document(page_content=excel_text)]
            else:
                # Fallback loader for other file types
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()
            
            chunks = self.text_splitter.split_documents(documents)
            return [chunk.page_content for chunk in chunks]
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            # Try fallback text extraction if normal loading fails
            try:
                if file_path.lower().endswith('.pdf'):
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    return [text]
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    return [self._excel_to_text(file_path)]
            except Exception as fallback_error:
                logger.error(f"Fallback extraction failed: {str(fallback_error)}")
            
            raise Exception(f"Could not process document: {str(e)}")

    def upload_documents(self, uploaded_files) -> dict:
        """Process and store uploaded documents for later querying."""
        try:
            # Clear previous documents
            self.clear_documents()
            
            # Process each uploaded file
            all_chunks = []
            for uploaded_file in uploaded_files:
                if uploaded_file.size > 20 * 1024 * 1024:  # 20MB limit
                    continue
                
                # Check file extension
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext not in ['.pdf', '.txt', '.text', '.xlsx', '.xls']:
                    continue
                
                # Save the uploaded file temporarily
                file_path = self._save_uploaded_file(uploaded_file)
                self.uploaded_documents.append(file_path)
                
                # Process the document
                chunks = self._process_document(file_path)
                all_chunks.extend(chunks)
            
            # Create vector store from document chunks
            if all_chunks:
                self.vector_store = Chroma.from_texts(
                    texts=all_chunks,
                    embedding=self.embeddings,  # Use our embedding class
                    persist_directory=self.documents_dir
                )
                self.vector_store.persist()
            
            return {
                "status": "success",
                "message": f"Processed {len(self.uploaded_documents)} document(s)",
                "documents": [os.path.basename(doc) for doc in self.uploaded_documents]
            }
        except Exception as e:
            logger.error(f"Error in upload_documents: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _format_docs(self, docs):
        """Format documents for display in the response."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query_documents(self, question: str, request=None) -> dict:
        """Query uploaded documents with context-aware resolution and entity tracking."""
        try:
            # Load vector store if not already loaded
            if not self.vector_store and os.path.exists(self.documents_dir):
                self.vector_store = Chroma(
                    persist_directory=self.documents_dir,
                    embedding_function=self.embeddings
                )
                
            if not self.vector_store:
                return {
                    "type": "error",
                    "message": "Please upload documents before querying."
                }
            
            # Step 1: Resolve references in question using conversation context
            resolved_question = self._resolve_document_query(question, request)
            
            # Step 2: Retrieve relevant document chunks
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            relevant_docs = retriever.get_relevant_documents(resolved_question)
            
            # Step 3: Extract entities from found documents
            if hasattr(self, 'entity_store'):
                for doc in relevant_docs:
                    self.entity_store.add_text(
                        doc.page_content,
                        {
                            'source': 'document',
                            'user_id': request.user.id if request else None,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
            
            # Step 4: Prepare enhanced prompt with context
            template = """Answer the question based on the document context and conversation history:
            
            Document Context:
            {context}
            
            Conversation Context:
            {conversation_context}
            
            Guidelines:
            1. For Excel data: Provide exact values when available
            2. For comparisons: Highlight differences clearly
            3. For follow-up questions: Maintain reference to previous entities
            4. If unsure: Say "This information isn't in the documents"
            
            Question: {question}"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Get conversation context if available
            conversation_context = ""
            if request and hasattr(self, 'entity_store'):
                last_entities = self.entity_store.get_recent_entities(
                    request.user.id if request else None,
                    limit=3
                )
                if last_entities:
                    conversation_context = "Recent references: " + ", ".join(last_entities)
            
            # Create RAG chain
            rag_chain = (
                {
                    "context": lambda x: self._format_docs(relevant_docs),
                    "conversation_context": lambda x: conversation_context,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Execute query
            response = rag_chain.invoke(resolved_question)
            
            # Step 5: Store entities from response
            if hasattr(self, 'entity_store'):
                self.entity_store.add_text(
                    response,
                    {
                        'user_id': request.user.id if request else None,
                        'source': 'document_response',
                        'is_response': True
                    }
                )
            
            return {
                "type": "text",
                "data": response
            }
            
        except Exception as e:
            logger.error(f"Document query error: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "message": f"Document query failed: {str(e)}"
            }

    def _resolve_document_query(self, question: str, request) -> str:
        """Resolve references in document queries using conversation context."""
        if not hasattr(self, 'entity_store') or not request:
            return question
        
        # Get last 5 messages for context
        context_messages = []
        if hasattr(self, 'get_chat_history_manager'):
            history_manager = self.get_chat_history_manager(request)
            if history_manager:
                context_messages = history_manager.get_context_messages(limit=5)
        
        # Resolve pronouns and references
        if hasattr(self.entity_store, 'resolve_reference'):
            resolved = self.entity_store.resolve_reference(question)
            if resolved:
                question = re.sub(
                    r'\b(it|they|this|that|those)\b',
                    resolved,
                    question,
                    flags=re.IGNORECASE
                )
        
        # Handle implicit follow-ups (e.g., "What about the deadline?")
        if self._is_followup_question(question):
            last_entity = None
            if hasattr(self.entity_store, 'get_last_entity'):
                last_entity = self.entity_store.get_last_entity(
                    request.user.id if request else None
                )
            elif hasattr(self.entity_store, 'get_recent_entities'):
                recent = self.entity_store.get_recent_entities(
                    request.user.id if request else None,
                    limit=1
                )
                if recent:
                    last_entity = recent[0]
            
            if last_entity:
                question = f"{last_entity} {question}"
        
        return question

    def _is_followup_question(self, question: str) -> bool:
        """Detect if question is likely a follow-up needing context."""
        question = question.lower().strip()
        return (
            len(question.split()) <= 6 and
            any(q_word in question for q_word in ['what', 'when', 'where']) and
            not any(term in question for term in ['document', 'file', 'page'])
        )
        
    def clear_documents(self):
        """Clear all uploaded documents and reset the vector store."""
        try:
            # Delete temporary document files
            for doc_path in self.uploaded_documents:
                if os.path.exists(doc_path):
                    os.remove(doc_path)
            
            # Reset vector store
            self.vector_store = None
            self.uploaded_documents = []
            
            # Clear ChromaDB persistence directory
            if os.path.exists(self.documents_dir):
                for filename in os.listdir(self.documents_dir):
                    file_path = os.path.join(self.documents_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
            
            return {
                "status": "success",
                "message": "All documents cleared"
            }
        except Exception as e:
            logger.error(f"Error in clear_documents: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }