PROMPT_NR = """You are an image quality expert.
Briefly describe the dominant degradations (e.g., blur, noise, exposure) in one sentence.
Then pick ONE quality bin in steps of 0.5.

Answer format:
[reasoning up to 64 tokens]
FinalBin: <low>-<high>
"""
def format_prompt():
    return PROMPT_NR
