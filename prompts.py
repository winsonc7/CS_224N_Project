def default_nego_agent_prompt():
    return f"""You are an expert negotiator. Given a negotiation conversation, provide advice for the seller on how to proceed.
    """

def default_eval_agent_prompt():
    return f"""Rate the quality of the following negotiation advice on a scale of 0 to 10, considering clarity, effectiveness, and professionalism.
    <Important>
    Format the feedback as 'Rating:X' without the commas or other punctuations. If no advice is provided, output 'Rating:0'.
    </Important>
    """

def compare_simple_eval_agent_prompt():
    return f"""You are given two samples of negotiation advice, labeled Sample 1
    and Sample 2. Output a single number, 1 or 2, corresponding to which sample
    provides better advice, considering clarity, effectiveness, and professionalism.
    """

def cot_eval_agent_prompt():
    return f"""Analyze the quality of the following negotiation advice, considering
    clarity, effectiveness, and professionalism. Based on your analysis, state the quality level of the advice and support your answer with reasoning and evidence. Finally, rate the quality of the negotiation advice on a scale of 0 to 10, considering clarity, effectiveness, and professionalism.
    <Important>
    Format the feedback as 'Rating:X' without the commas or other punctuations. If no advice is provided, output 'Rating:0'. Make sure this is the final part of your output.
    </Important>
    """