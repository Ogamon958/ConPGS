# Controllable Paraphrase Generation on Semantic and Lexical Similairties

## Release Similarity-Controllable Paraphrase Generation (SCPG) model  
SCPG model (Used in the paper) -> https://huggingface.co/Ogamon/scpg_model  
SCPG model ver2 (Error Corrected Version) -> https://huggingface.co/Ogamon/scpg_model_ver2  


## Paraphrase Corpora  
https://drive.google.com/drive/folders/1V96SiVkgzlW9bn98K3S0q968vfoTOPy7?usp=sharing  
(Constructing paraphrase corpora from wiki40b with SCPG model)


## How to use SCPG model

```
#setup
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('Ogamon/scpg_model')
tokenizer = BartTokenizer.from_pretrained('Ogamon/scpg_model')
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

#tags
sim_token = {70:"<SIM70>", 75:"<SIM75>",80:"<SIM80>",85:"<SIM85>",90:"<SIM90>",95:"<SIM95>"}
bleu_token={5:"<BLEU0_5>",10:"<BLEU10>",15:"<BLEU15>",20:"<BLEU20>",25:"<BLEU25>",30:"<BLEU30>",35:"<BLEU35>",40:"<BLEU40>"}


```
#edit here
text = "The tiger sanctuary has been told their 147 cats must be handed over."
sim = sim_token[95] #70,75,80,85,90,95
bleu = bleu_token[5] #5,10,15,20,25,30,35,40 


#evaluate
model.eval()
with torch.no_grad():
    input_text=f"{sim} {bleu} {text}"  
    inputs = tokenizer.encode(input_text, return_tensors="pt",truncation=True).to(device)
    length=inputs.size()[1]
    max_len=int(length*1.5)
    min_len=int(length*0.75)       
    
    summary_ids = model.generate(inputs,max_length=max_len,min_length=min_len,num_beams=5)
    summary = tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    print(summary)
    #The tiger sanctuary was told to hand over its 147 cats.
```
