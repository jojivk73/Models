import gradio as gr
import argparse

from myutils import *
from hf_zephyr import *
from google_search import google_it
import sys

from constants_and_prompt import *

parser = argparse.ArgumentParser('Intel RAG')
parser.add_argument('-m', '--model', default=model_name, required=False) 
parser.add_argument('-w', '--web_link', default=search_str, required=False) 
parser.add_argument('-s', '--simple_prompt', default=True, required=False) 
args = parser.parse_args()
if args.model:
  model_name = args.model
if args.web_link:
  search_str = args.web_link

web_links= (search_str,)
print(" Web Link:", web_links)
vectorstore = read_vectorize_links(web_links, modelPath,
                    model_kwargs, encode_kwargs, db_path)
LLM,tokenizer = get_hf_zephyr_model(model_name)
rag_prompt =build_prompt(tokenizer)

def get_answer(rag_prompt, quest):
    #print("Query :", quest)
    docs = vectorstore.similarity_search(quest)
    final_prompt = generate_final_prompt(docs, rag_prompt, quest)
    print("Information: Answer being generated...!")
    #answers = LLM(final_prompt)[0]["generated_text"]
    answers = LLM(final_prompt)
    return answers

loc_prompt = instr_prompt_simple if (args.simple_prompt) else instr_prompt
answers = get_answer(rag_prompt, loc_prompt)
for i in range(len(answers)):
    print("-----------------------------------------------")
    print(" Answer :", answers[i]["generated_text"])
#for pro in processsors:
#  np = prompts[1].format(pro, processors)
#  get_answer(rag_prompt, np)
