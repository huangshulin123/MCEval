This is a repository for the paper ***MCEval: A Dynamic Framework for Fair Multilingual Cultural Evaluation of LLMs***. 

We propose MCEval, a novel multilingual evaluation framework that employs the dynamic cultural question construction and enables causal analysis through Counterfactual Rephrasing and Confounder Rephrasing. Our comprehensive evaluation spans 13 cultures and 13 languages, systematically assessing both cultural awareness and cultural bias across different linguistic scenarios.

In this repository, we provide the code. 

### Workflow
The workflow demonstrates how to dynamically generate data from the original cultural information and evaluate LLM performance:

1. **Step 1: Construct Data**  
   Run the following command to generate the Original Cultural Questions, Counterfactual Questions and Confounder Questions, based on the original cultural information:

   ```bash
   python workflow/construct_data.py
   ```

2. **Step 2: Translate Data**  
   Execute this script to translate Original Cultural Questions, Counterfactual Questions and Confounder Questions:

   ```bash
   python workflow/translate.py --Language French --bsz 5 --mode bias
   ```

   You can replace `French` with other language and adjust the batch size (`--bsz`) as needed. We provide two modes: `bias` and `awareness` for cultural analysis.

3. **Step 3: Merge the Translations**  
   Use this command to merge the translations:

   ```bash
   python workflow/merge_languages.py
   ```

4. **Step 4: Evaluate Model Performance**  
   Use this command to test the model on the dynamically generated cultural data (example using meta-llama/Llama-3.3-70B-Instruct):

```bash
python workflow/test_LLM.py --Eval_model meta-llama/Llama-3.3-70B-Instruct --mode bias
```

You can replace `meta-llama/Llama-3.3-70B-Instruct` with other model names and adjust the(`--mode`) as needed.