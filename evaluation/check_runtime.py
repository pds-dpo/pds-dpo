"""Read-only runtime check; no model loading or API requests."""
import importlib.metadata
import argparse
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['inference','scoring'],required=True);args=parser.parse_args()
    expected={'transformers':'4.57.6','datasets':'4.4.2','peft':'0.18.1','accelerate':'1.12.0'} if args.profile=='inference' else {'numpy':'1.26.4','spacy':'3.7.2','nltk':'3.8.1','openai':'3.1.0'}
    for package,version in expected.items():
        actual=importlib.metadata.version(package)
        if actual!=version: raise RuntimeError(f'{package}: expected {version}, found {actual}')
        print(package,actual)
    if args.profile=='inference':
        sys.path.insert(0,str(Path(__file__).resolve().parent/'compat'))
        from llava.model.builder import load_pretrained_model
        import torch
        print('LLaVA import OK; torch',torch.__version__,'CUDA available',torch.cuda.is_available())
    else:
        import spacy
        spacy.load('en_core_web_lg')
        print('AMBER language model available')

if __name__=='__main__': main()
