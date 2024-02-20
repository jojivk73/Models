import gradio as gr

from myutils import *
from hf_zephyr import *
from google_search import google_it

from constants_and_prompt import *

LLM,tokenizer = get_hf_zephyr_model(model_name)
vecstore = None
rag_prompt=instr_prompt
def search_build_data(web_links):
  global vecstore
  vecstore = read_vectorize_links(web_links, modelPath,
                    model_kwargs, encode_kwargs, db_path)
  rag_prompt =build_prompt(tokenizer)
  docs = vecstore.similarity_search("Intel")
  #print("Information:", docs[0].page_content)
  return rag_prompt

def query_search(rag_prompt, question):
  global vecstore
  question = question +" :"+ instr_prompt
  docs = vecstore.similarity_search(question)
  final_prompt = generate_final_prompt(docs, rag_prompt, question)
  print("Information: Generating data for you be patient.....!")
  answer = LLM(final_prompt)[0]["generated_text"]
  return answer

history =[]
with gr.Blocks() as demo:
    error_box = gr.Textbox(label="Error", visible=False)

    search_box = gr.Textbox(label="Search")
    choice_box = gr.Radio(["Search Terms", "WebLink",])
    search_btn = gr.Button("Google")

    with gr.Row(visible=False) as output_row:
      with gr.Column(visible=False) as output_col:
        query_box = gr.Textbox(label="Enter a Query to Chat")
        query_btn = gr.Button("Query")
      answers_box = gr.Textbox(label="Answers")
      #Chat = gr.ChatInterface(google_it)

    def search(search, choice):
        global rag_prompt
        if len(search) == 0:
            return {error_box: gr.Textbox(value="Enter Search Terms or Link", visible=True)}

        if 'Search Terms' in choice:
          web_links = google_it(search, history)
        else :
          web_links= (search,)
        print("----------------------------------------------------")
        print("Start search,...!", web_links)
        print("----------------------------------------------------")
        rag_prompt=search_build_data(web_links)
        return {
            output_row: gr.Row(visible=True),
            output_col: gr.Column(visible=True),
            answers_box: search,
            #Chat :"?",
        }

    def query(new_query, answers):
        global rag_prompt
        answers = query_search(rag_prompt, new_query)
        return {
            output_row: gr.Row(visible=True),
            output_col: gr.Column(visible=True),
            answers_box: answers,
        }

    search_btn.click(
        search,
        [search_box, choice_box],
        [error_box, answers_box, output_row, output_col],
    )
    query_btn.click(
        query,
        [query_box, answers_box],
        [error_box, answers_box, output_row, output_col],
    )

demo.launch(share=True)
