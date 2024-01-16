"""
questionAnsweringPipeline.py

author: Moneera Alsuwailm


This script sets up and runs a question-answering pipeline using a language model and a retrieval system.
The pipeline uses a local Llama-2-7B-Chat model for language understanding, HuggingFace's sentence embeddings for vector representation,
and a FAISS vector store for efficient similarity search. The script takes a question as input and outputs the relevant answer along with
the source documents.

Dependencies:
  - torch
  - glob
  - langchain
  - nltk
  - rouge_score
  - bert_score

Usage:
  python questionAnsweringPipeline.py "your question here"
"""
import torch
import os
import glob
import argparse
import timeit
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import CTransformers


class QuestionAnsweringPipeline:
    """Class to encapsulate the question-answering pipeline."""

    def __init__(self, model_path, embeddings_model, vectorstore_path):
        self.llm = self._setup_language_model(model_path)
        self.vectordb = self._setup_vector_store(embeddings_model, vectorstore_path)
        self.qa_prompt = self._set_qa_prompt()
        self.dbqa = self._build_retrieval_qa()

    @staticmethod
    def _setup_language_model(model_path):
        """Initializes the language model with provided configuration."""
        return CTransformers(model=model_path, model_type='llama', config={
            'max_new_tokens': 512, 'temperature': 0.01
        })

    @staticmethod
    def _setup_vector_store(embeddings_model, vectorstore_path):
        """Sets up the FAISS vector store for document retrieval."""
        # Use GPU for embeddings if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device': device})
        return FAISS.load_local(vectorstore_path, embeddings)

    @staticmethod
    def _set_qa_prompt():
        """Defines and returns the prompt template for question-answering."""
        qa_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """
        return PromptTemplate(template=qa_template, input_variables=['context', 'question'])

    def _build_retrieval_qa(self):
        """Builds the RetrievalQA object using the language model and vector store."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.vectordb.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.qa_prompt}
        )

    def answer_question(self, question):
        """Takes a question and generates an answer using the RetrievalQA pipeline.

            Args:
                question (str): The question to be answered.

            Returns:
                dict: The response containing the answer and source documents.
        """

        return self.dbqa({'query': question})

    def update_vector_store(self, directory_path, vectorstore_path, embeddings_model):
        """
        Creates or updates the FAISS vector store based on the PDF files in the specified directory.

        Args:
            directory_path (str): Path to the directory containing PDF files.
            vectorstore_path (str): Path to the FAISS vector store.
            embeddings_model (str): Model name for HuggingFace embeddings.
        """
        # Ensure the vectorstore directory exists
        if not os.path.exists(vectorstore_path):
            os.makedirs(vectorstore_path)

        # Check if there are PDFs in the data_test directory
        if not glob.glob(os.path.join(directory_path, "*.pdf")):
            raise FileNotFoundError("No PDF files found in the specified directory.")

        # Check if vector store already exists and if the directory has changed
        if self._should_update_vector_store(directory_path, vectorstore_path):
            documents = self._load_pdf_documents(directory_path)
            if documents:
                texts = self._split_documents(documents)
                # Use GPU for embeddings if available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device': device})
                vectorstore = FAISS.from_documents(texts, embeddings)
                vectorstore.save_local(vectorstore_path)
                print("Vector store created or updated.")
            else:
                print("No PDFs found. Vector store not updated.")
        else:
            print("No update needed for vector store.")

    @staticmethod
    def _load_pdf_documents(directory_path):
        """Loads PDF documents from a specified directory."""
        loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
        return loader.load()

    @staticmethod
    def _split_documents(documents):
        """Splits text documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    @staticmethod
    def _should_update_vector_store(directory_path, vectorstore_path):
        """Determines whether the vector store needs to be updated."""
        existing_files = set(glob.glob(os.path.join(directory_path, "*.pdf")))
        try:
            with open(os.path.join(vectorstore_path, 'file_list.txt'), 'r') as file:
                stored_files = set(file.read().splitlines())
        except FileNotFoundError:
            stored_files = set()

        # Update file list for future checks
        with open(os.path.join(vectorstore_path, 'file_list.txt'), 'w') as file:
            file.write('\n'.join(existing_files))

        return existing_files != stored_files


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Question Answering Pipeline")
    parser.add_argument('question', type=str, help="Question to be answered")
    args = parser.parse_args()

    # Initialize and run the QA pipeline
    try:
        start = timeit.default_timer()
        qa_pipeline = QuestionAnsweringPipeline(
            model_path='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            vectorstore_path='vectorstore/db_faiss'
        )

        # Update vector store if necessary
        qa_pipeline.update_vector_store(
            directory_path='data_test/',
            vectorstore_path='vectorstore/db_faiss',
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2"
        )

        response = qa_pipeline.answer_question(args.question)
        end = timeit.default_timer()

        # Display answer and source documents
        print(f'\nAnswer: {response["result"]}')
        print('=' * 50)
        for i, doc in enumerate(response['source_documents']):
            print(f'\nSource Document {i + 1}\n')
            print(f'Source Text: {doc.page_content}')
            print(f'Document Name: {doc.metadata["source"]}')
            print(f'Page Number: {doc.metadata["page"]}\n')
            print('=' * 50)

        print(f"Time to retrieve response: {end - start:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")
