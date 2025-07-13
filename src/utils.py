from src.models import LLMModel
paraphraser = LLMModel("gpt-4o", max_new_tokens=15000, temperature=0.7)

def Verifier_translate(translated_question,original_question,language):
    prompt = """Determine whether the translated question below is reasonable. The criteria for judgment are: reasonable, does not alter the core meaning of the original question, and adheres to translation norms in terms of wording.
    
Target Translated Language: {language}

Original Question: {original_question}

Translated Question: {translated_question}

If the translated question is reasonable, strictly provide <<<Yes>>>; if the translated question is not reasonable, strictly provide <<<No>>>."""
    dic = {"translated_question":translated_question,"original_question":original_question,"language":language}
    temp = prompt.format(**dic)
    out = paraphraser(temp)
    if "<<<Yes>>>" in out:
        return True
    else:
        return False

def Verifier_generator(sent_dict):
    prompt = """
    Evaluate whether the following Cultural Awareness Question, constructed based on Cultural Tradition, is reasonable. Assess whether the question preserves the core content of the original Cultural Tradition and whether the constructed Cultural Awareness Question includes a Cultural Behavior. Note that the Cultural Behavior here should reflect the Cultural Tradition and demonstrate an indirect relationship, preferably a causal one, where Cultural Tradition is the cause and Cultural Behavior is the effect, rather than a direct or simple rewrite of the Cultural Tradition. Additionally, the Cultural Awareness Question must include the corresponding causal and irrelevant items.

Cultural Tradition: {awareness_info}

Cultural Behavior: {behavior}

Cultural Awareness Question: {original_question}

If the constructed Cultural Awareness Question is reasonable, strictly output <<<Yes>>>. If the constructed Cultural Awareness Question is unreasonable, strictly output <<<No>>>."""
    temp = prompt.format(**sent_dict)
    out = paraphraser(temp)
    if "<<<Yes>>>" in out:
        return True
    else:
        return False
def Verifier_rephrase(sent_dict):
    prompt = """Evaluate whether the following rephrasing of the question are reasonable. We obtained the Causal Rewrite of the Question and Confounding Rewrite of the Question by applying Causal Rewriting and Confounding Rewriting to the Original Question. Note that Causal Rewriting refers to changing the direction of the Cultural Awareness Question's answer by altering the cause, while Confounding Rewriting should not change the direction of the Cultural Awareness Question's answer. Ensure that Counterfactual Rephrasing changes the answer's direction, whereas Confounder Rephrasing does not.

Original Question: {original_question}
    
Causal Rewrite of the Question: {counterfactual_question}
    
Confounding Rewrite of the Question: {confounding_question}
    
If both rephrasings are reasonable, strictly respond with <<<Yes>>>; if either rephrasing is unreasonable, strictly respond with <<<No>>>."""
    temp = prompt.format(**sent_dict)
    out = paraphraser(temp)
    if "<<<Yes>>>" in out:
        return True
    else:
        return False
