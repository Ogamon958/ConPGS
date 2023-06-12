# similarity-controllable-paraphrase-generation


hugging face model -> https://huggingface.co/Ogamon/scpg_model
hugging face ver2 model -> https://huggingface.co/Ogamon/scpg_model_ver2

dataset -> https://drive.google.com/drive/folders/1V96SiVkgzlW9bn98K3S0q968vfoTOPy7?usp=sharing


## how to use

```
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('Ogamon/scpg_model')
tokenizer = BartTokenizer.from_pretrained('Ogamon/scpg_model')
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sim_token = {70:"<SIM70>", 75:"<SIM75>",80:"<SIM80>",85:"<SIM85>",90:"<SIM90>",95:"<SIM95>"}
bleu_token={05:"<BLEU0_5>",10:"<BLEU10>",15:"<BLEU15>",20:"<BLEU20>",25:"<BLEU25>",30:"<BLEU30>",35:"<BLEU35>",40:"<BLEU40>"}
```


```
#edit here
text = "I usually play video games for two hours."
sim = sim_token[70] #70,75,80,85,90,95
bleu = bleu_token[05] #05,10,15,20,25,30,35,40 

input_text=f"{sim} {bleu} {text}"  
inputs = tokenizer.encode(text, return_tensors="pt",truncation=True).to(device)

#evaluate
model.eval()
with torch.no_grad():
    summary_ids = model.generate(inputs,max_new_tokens=128) #
    summary = tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    print(summary)
```
