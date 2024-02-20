
import torch
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
if torch. cuda. is_available():
  model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': False}
#db_path = "/localdisk/jojimon/RAG/db/lancedb"
db_path = "/tmp/IntelRAG/db/lancedb"
model_name = "HuggingFaceH4/zephyr-7b-beta"
#model_name = "HuggingFaceM4/idefics-9b"
#model_name = "HuggingFaceM4/idefics-9b-instruct"
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tok_max_length=1024
splitter_chunk=3000
splitter_overlap=700
single_link =True

#search_str ="Intel CPU performance"
search_str='https://www.pcmag.com/news/meteor-lake-first-tests-intel-core-ultra-7-benched-for-cpu-graphics-and'
default_link =search_str
#question ='Generate a table of processor performance comparisons that are in the doc'

instr_prompt= 'Do the following instructions step by step.' + \
'Step 1. List the processors in the document.' + \
'Step 2. For each processor compare its performance with other processors in the document\n' + \
'Step 3. Get the performance number for each processor:\n' + \
' Example : Cinebench R23 - Multi-core CPU Score   H   14476\n' + \
'Step 4. Generate a table as below for processors and their performance numbers\n' + \
' Example table : \n' + \
'Benchmark	Type	Metric	H/L	Acer Swift Go 14	Asus Zenbook 14	MSI Prestige 16	i7-1360P	R7-7840U	Apple M2	i7-1370P	Snapdragon X Elite	Lenovo X1 Carbon\n' + \
'Sapphire Rapids - Multi-core	CPU	Score	H	1234			9283		810	\n'	 + \
'Cinebench R24 - Single Core	CPU	Score	H	12			1039		121		152 \n'	 + \
'Grand Bench 6.2 - Multi-core	CPU	Score	H	123			111		10222		12223 \n' 

