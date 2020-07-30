import argparse
import tarfile
import io
import json
import pickle as pkl
import os
import collections
import torch
import pandas as pd 
import stanfordnlp
import re


def filt(x):
    if len(str(x).split())>100:
        return x
    else:
        return 'NA'

def filt2(x):
    if len(str(x).split())>10:
        return x
    else:
        return 'NA'

def main(args):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    en_nlp = stanfordnlp.Pipeline(lang='en',processors='tokenize,mwt')
    data=pd.read_csv(args.path)
    pat=re.compile('tl....dr|tl..dr|tl.dr|tldr')
    new_data={}
    new_data['normalized_body']=data['normalized_body'].apply(lambda x: str(x).lower())
    new_data['summary']=data['summary'].apply(lambda x: str(x).lower())
    new_data['normalized_body']=new_data['normalized_body'].apply(lambda x:pat.split(str(x))[0] )
    new_data['summary']=new_data['summary'].apply(lambda x: pat.split(str(x))[0])
    new_data['summary']=data['summary'].apply(lambda x: str(x).lower())
    new_data['normalized_body']=new_data['normalized_body'].apply(lambda x: filt(x))
    new_data['summary']=new_data['summary'].apply(lambda x: filt2(x))
    new_data=pd.DataFrame(new_data)
    filter_new_Data=new_data.query('normalized_body!="NA"')
    filter_new_Data=filter_new_data.query('summary!="NA"')
    filter_new_Data.head()
    vocab_counter = collections.Counter()
    en_nlp = stanfordnlp.Pipeline(lang='en',processors='tokenize,mwt')
    
    index=0
    
    with tarfile.open('train', 'w') as writer:
      for x,y in zip(filter_new_Data['normalized_body'][:900000],filter_new_Data['summary'][:900000]):
          index+=1
          article=[]
          summary=[]
          art=en_nlp(x)
          suma=en_nlp(y) 
          for i in art.sentences:
            article.append(" ".join([j.text for j in i.tokens]))
          for k in suma.sentences:
            summary.append(" ".join([m.text for m in k.tokens]))
          js_example = {}
          js_example['id'] = index
          js_example['article'] = article
          js_example['abstract'] = summary
          js_serialized = json.dumps(js_example, indent=4).encode()
          save_file = io.BytesIO(js_serialized)
          tar_info = tarfile.TarInfo('{}/{}.json'.format(os.path.basename('train'), index))
          tar_info.size = len(js_serialized)
          writer.addfile(tar_info, save_file)
          if index%1000==0:
              print(index)
          art_tokens = ' '.join(article).split()
          abs_tokens = ' '.join(summary).split()
          tokens = art_tokens + abs_tokens
          tokens = [t.strip() for t in tokens] # strip
          tokens = [t for t in tokens if t != ""] # remove empty
          vocab_counter.update(tokens)
    
    print("Writing vocab file...")
    with open( "vocab_cnt.pkl",'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)
    print("Finished writing vocab file")
    
    
    index=0
    with tarfile.open('val', 'w') as writer:
      for x,y in zip(filter_new_Data['normalized_body'][900000:1000000],filter_new_Data['summary'][900000:1000000]):
          index+=1
          article=[]
          summary=[]
          art=en_nlp(x)
          suma=en_nlp(y) 
          for i in art.sentences:
            article.append(" ".join([j.text for j in i.tokens]))
          for k in suma.sentences:
            summary.append(" ".join([m.text for m in k.tokens]))
          js_example = {}
          js_example['id'] = index
          js_example['article'] = article
          js_example['abstract'] = summary
          js_serialized = json.dumps(js_example, indent=4).encode()
          save_file = io.BytesIO(js_serialized)
          tar_info = tarfile.TarInfo('{}/{}.json'.format(os.path.basename('val'), index))
          tar_info.size = len(js_serialized)
          writer.addfile(tar_info, save_file)
          if index%1000==0:
              print(index)
    
    index=0
    with tarfile.open('test', 'w') as writer:
      for x,y in zip(filter_new_Data['normalized_body'][1000000:],filter_new_Data['summary'][1000000:]):
          index+=1
          article=[]
          summary=[]
          art=en_nlp(x)
          suma=en_nlp(y) 
          for i in art.sentences:
            article.append(" ".join([j.text for j in i.tokens]))
          for k in suma.sentences:
            summary.append(" ".join([m.text for m in k.tokens]))
          js_example = {}
          js_example['id'] = index
          js_example['article'] = article
          js_example['abstract'] = summary
          js_serialized = json.dumps(js_example, indent=4).encode()
          save_file = io.BytesIO(js_serialized)
          tar_info = tarfile.TarInfo('{}/{}.json'.format(os.path.basename('test'), index))
          tar_info.size = len(js_serialized)
          writer.addfile(tar_info, save_file)
          if index%1000==0:
              print(index)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preaparing data for the model'
    )
    parser.add_argument('--path', required=True, help='path csv file (file included)')

    args = parser.parse_args()

    main(args)

