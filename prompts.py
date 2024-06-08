def reward_model_prompt():
    return f"""You are an expert negotiator. You will be provided a negotiation conversation, as well as advice for that negotiation. Given the context, first rate the quality of the advice on a scale of 1 to 10 in four categories:
        1. How professional is the advice? (higher score = more professional)
        2. How specific is the advice to the conversation? (higher score = more specific)
        3. How effective does the advice seem? (higher score = more effective)
        4. How concise is the advice? (higher score = more concise)
    Lastly, output your holistic rating of the advice as the average of the four scores.
    <Important>
    Format your final rating as 'Rating:X' without the commas or other punctuations. If no advice is provided, output 'Rating:0'. 
    </Important>"""

def default_nego_agent_prompt():
    return f"""You are an expert sales negotiator. Given the following negotiation conversation, provide helpful advice to the seller on how to proceed.
    """

def cot_nego_agent_prompt():
    return f"""You are an expert sales negotiator. Given the following negotiation conversation, provide helpful advice to the seller on how to proceed. Explain your reasoning behind your suggestions.
    """

def rating_eval_agent_prompt():
    return f"""You are an expert negotiator. You will be provided a negotiation conversation, as well as advice for that negotiation. Given the context, your task is to rate the quality of the advice on a scale of 1 to 10 in four categories:
            1. How professional is the advice? (higher score = more professional)
            2. How specific is the advice to the conversation? (higher score = more specific)
            3. How effective does the advice seem? (higher score = more effective)
            4. How concise is the advice? (higher score = more concise)
        Output your ratings as a list of four numbers. Example: '[8, 2, 10, 4]'
    """

def compare_eval_agent_prompt():
    return f"""You are an expert negotiator. You will be provided a negotiation conversation, as well as two pieces of advice for that negotiation labeled Sample 1 and Sample 2. Output a single number, 1 or 2, corresponding to which sample provides better advice.
    """

def compare_eval_agent_verbose_prompt():
    return f"""You are an expert negotiator. You will be provided a negotiation conversation, as well as two pieces of advice for that negotiation labeled Sample 1 and Sample 2. Output which sample provides better advice, and justify your choice with reasoning and explanation.
    """
