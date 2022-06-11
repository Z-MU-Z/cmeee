from transformers import AlbertTokenizer, BertModel
tokenizer = AlbertTokenizer.from_pretrained('uer/roberta-base-word-chinese-cluecorpussmall')
# tokenizer.add_tokens(['℃'])
model = BertModel.from_pretrained("uer/roberta-base-word-chinese-cluecorpussmall")
text = "我是DNA，也就是脱氧核糖核苷酸℃！"
tokenized_input = tokenizer.tokenize(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
