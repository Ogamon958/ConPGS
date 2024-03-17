# Controllable Paraphrase Generation for Semantic and Lexical Similarities

## Release Paraphrase Generation model (conpgs_model) 

https://huggingface.co/Ogamon/conpgs_model  


## Paraphrase Corpora  
https://drive.google.com/drive/folders/1V96SiVkgzlW9bn98K3S0q968vfoTOPy7?usp=sharing  
(Constructing paraphrase corpora from wiki40b with our model)


## How to use our Paraphrase Generation model

```
#setup
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('Ogamon/conpgs_model')
tokenizer = BartTokenizer.from_pretrained('Ogamon/conpgs_model')
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

#tags
sim_token = {70:"<SIM70>", 75:"<SIM75>",80:"<SIM80>",85:"<SIM85>",90:"<SIM90>",95:"<SIM95>"}
bleu_token={5:"<BLEU0_5>",10:"<BLEU10>",15:"<BLEU15>",20:"<BLEU20>",25:"<BLEU25>",30:"<BLEU30>",35:"<BLEU35>",40:"<BLEU40>"}
```

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


## How to use our Paraphrase Generation model

```
#setup
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('Ogamon/conpgs_model')
tokenizer = BartTokenizer.from_pretrained('Ogamon/conpgs_model')
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

#tags
sim_token = {70:"<SIM70>", 75:"<SIM75>",80:"<SIM80>",85:"<SIM85>",90:"<SIM90>",95:"<SIM95>"}
bleu_token={5:"<BLEU0_5>",10:"<BLEU10>",15:"<BLEU15>",20:"<BLEU20>",25:"<BLEU25>",30:"<BLEU30>",35:"<BLEU35>",40:"<BLEU40>"}
```


## Citation
Please cite our LREC-COLING2024 paper if you use this repository:

```
@inproceedings{ogasa-2024-lrec-coling,
    title = {{Controllable Paraphrase Generation for Semantic and Lexical Similarities}},
    author = "Ogasa, Yuya  and Kajiwara, Tomoyuki and Arase, Yuki",
    booktitle = {The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
    year = "2024",
}
```
