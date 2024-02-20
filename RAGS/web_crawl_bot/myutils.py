import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import WebBaseLoader, HuggingFaceDatasetLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import LanceDB
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

import lancedb

from constants_and_prompt import *
#import nest_asyncio

def load_web_links(web_links):
  #nest_asyncio.apply()
  loader = WebBaseLoader(
    web_paths=web_links,
    #bs_kwargs=dict(
    #    parse_only=bs4.SoupStrainer(
    #        # JMJ What is this. How to improve this?
    #        class_=("post-content", "post-title", "post-header")
    #    )
    #)
  )
  docs = loader.load()
  if (len(docs) > 0):
      print("Information: Docs loaded successfully..!")
  return docs

def data_splitter(docs, chunk=splitter_chunk, overlap=splitter_overlap):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
  #text_splitter = TokenTextSplitter(encoding_name=model_name, chunk_size=chunk, chunk_overlap=overlap)
  splits = text_splitter.split_documents(docs)
  return splits


def get_hf_embeddings(modelPath,m_kwargs, e_kwargs):
  embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,       # Provide the pre-trained model's path
        model_kwargs=m_kwargs,  # Pass the model configuration options
        encode_kwargs=e_kwargs # Pass the encoding options
  )
  return embeddings

def get_db(embeddings, db_path):
  db = lancedb.connect(db_path)
  table = db.create_table(
    "my_table",
     data=[
         {
             "vector": embeddings.embed_query("Hello World"),
             "text": "Hello World",
             "id": "1",
         }
     ],
     mode="overwrite",
  )
  return table, db


def get_vectorstore(splits, embeds, table):
  vectorstore = LanceDB.from_documents(documents=splits, embedding=embeds, connection=table)
  print("Information: Vector store created successfully..!")
  retriever = vectorstore.as_retriever()
  return vectorstore, retriever


def read_vectorize_links(web_links, modelPath, model_kwargs, encode_kwargs, db_path):
  print(' Reading Web links :', web_links[0])
  docs = load_web_links(web_links)
  splits = data_splitter(docs)
  embeddings = get_hf_embeddings(modelPath, model_kwargs, encode_kwargs)
  print("Information: Embeddings generated successfully..!")
  table, db = get_db(embeddings, db_path)

  vectorstore, retriever = get_vectorstore(splits, embeddings, table)

  return vectorstore


def test_similarity(vectorstore, question):
  searchDocs = vectorstore.similarity_search(question)
  print("----------------------------------------------")
  print(searchDocs[0].page_content)
  print("----------------------------------------------")


def load_model_for_qa(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=tok_max_length)
  model = AutoModelForQuestionAnswering.from_pretrained(model_name)

  return model, tokenizer

def my_pipeline(task, model_name, tokenizer, vectorstore):
  # Define a question-answering pipeline using the model and tokenizer
  question_answerer = pipeline(
    task,
    model=model_name,
    tokenizer=tokenizer,
    return_tensors='pt'
  )

  # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
  # with additional model-specific arguments (temperature and max_length)
  llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_length": 1024},
  )
  retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
  return qa
