# Controllable Paraphrase Generation for Semantic and Lexical Similarities

## <u>Con</u>trollable <u>P</u>araphrase <u>G</u>eneration for Semantic and Lexical <u>S</u>imilarities model (conpgs_model) 

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


## Citation
Please cite our LREC-COLING2024 paper if you use this repository:  
(Details updated after submission.)

```
@inproceedings{ogasa-2024-lrec-coling,
    title = {{Controllable Paraphrase Generation for Semantic and Lexical Similarities}},
    author = "Ogasa, Yuya  and Kajiwara, Tomoyuki and Arase, Yuki",
    booktitle = "The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "Association for Computational Linguistics",
    abstract = "We developed a controllable paraphrase generation model for semantic and lexical similarities using a simple and intuitive mechanism: attaching tags to specify these values at the head of the input sentence. Lexically diverse paraphrases have been long coveted for data augmentation. However, their generation is not straightforward because diversifying surfaces easily degrades semantic similarity. Furthermore, our experiments revealed two critical features in data augmentation by paraphrasing: appropriate similarities of paraphrases are highly downstream task-dependent, and mixing paraphrases of various similarities negatively affects the downstream tasks. These features indicated that the controllability in paraphrase generation is crucial for successful data augmentation. We tackled these challenges by fine-tuning a pre-trained sequence-to-sequence model employing tags that indicate the semantic and lexical similarities of synthetic paraphrases selected carefully based on the similarities. The resultant model could paraphrase an input sentence according to the tags specified. Extensive experiments on data augmentation for contrastive learning and pre-fine-tuning of pretrained masked language models confirmed the effectiveness of the proposed model. We release our paraphrase generation model and a corpus of 87 million diverse paraphrases. (https://github.com/Ogamon958/ConPGS)"
}
```
