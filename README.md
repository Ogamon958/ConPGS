# similarity-controllable-paraphrase-generation

how to use

'''
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('Ogamon/scpg_model')
tokenizer = BartTokenizer.from_pretrained('Ogamon/scpg_model')
'''


'''
sim_token = ["<SIM70>","<SIM75>","<SIM80>","<SIM85>","<SIM90>","<SIM95>"]
bleu_token=["<BLEU0_5>","<BLEU10>","<BLEU15>","<BLEU20>","<BLEU25>","<BLEU30>","<BLEU35>","<BLEU40>"]

text = "I usually play video games for two hours."
  
input_text=f"{sim_token[]} {bleu_token[]}"  

inputs = tokenizer.encode(text, return_tensors="pt",truncation=True).to(device)

# 推論
model.eval()
with torch.no_grad():
    summary_ids = model.generate(inputs,max_new_tokens=128)
    summary = tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    print(summary)
'''
