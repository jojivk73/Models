


This RAG reads a given web link and collects performance data for processors.
It would generate a taable of performance comparisions.For best results higher quality models need to be used.

Instructions
-------------

1. pip install -r requirement.txt
2. Open constants_and_prompt.py and change the db dir to point to a dir you want
3. Run the RAG.

   3.1 To run from directly in shell.
        ```
        python run_prompt.py.
        ```
      You can specify a web link with -w option.

   3.2 To run the web interface.
       ```
       python rag_ui.py.
       ```

Model used:
```  model_name = "HuggingFaceH4/zephyr-7b-beta"```

You can change the model used with the shell execution.
 ```> python run_prompt.py --model mistralai/Mistral-7B-Instruct-v0.2 ```

Limitation:
--------------------
Web search links hangs as some links are not valid.
Do not generate tables always.
Shell based execution generates tables more frequently.

Possible Improvements
--------------------------------
1. Use a large or better model like GPT-3.5 or Brad.
2. Filter out blocking links with web search.
3. Add advanced RAG processing like re-prioritization.
4. Fine-tune a medium size model to optimize generation of csv dataset.
