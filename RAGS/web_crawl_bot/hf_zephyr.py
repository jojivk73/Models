from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

def get_hf_zephyr_model(READER_MODEL_NAME):
  print("Information: Model loading please wait...!")
  model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
  if torch. cuda. is_available():
      model = model.to('cuda:0')
  print("Information: Model loaded...!")
  tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

  print("Information: Tokenizer created...!")
  READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
  )
  print("Information: Pipeline created...!")
  
  return READER_LLM, tokenizer


def build_prompt(tokenizer):
  prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
  give a comprehensive answer as a table to the question with data.
  Respond only to the question asked, response should be concise and relevant to the question.
  Provide the number of the source document when relevant.
  If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
  {context}
  ---
  Now here is the question you need to answer.

  Question: {question}""",
    },
  ]
  RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
      prompt_in_chat_format, tokenize=False, add_generation_prompt=True
  )
  return RAG_PROMPT_TEMPLATE



def generate_final_prompt(retrieved_docs, RAG_PROMPT_TEMPLATE, question):
  retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # we only need the text of the documents
  context = "\nExtracted documents:\n"
  context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

  final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
  print("Information: Final prompt generated...!")
  return final_prompt

if __name__ == '__main__' :
  LLM,tokenizer = get_hf_zephyr_model(READER_MODEL_NAME)
  rag_prompt =build_prompt(tokenizer)
  final_prompt = generate_final_prompt(docs, rag_prompt, "How to create a pipeline object?")
  answer = LLM(final_prompt)[0]["generated_text"]
  print(answer)

