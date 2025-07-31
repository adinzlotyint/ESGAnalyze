from definitions import BASE_MODEL_PATH, CONFIG
from transformers import AutoModel, AutoTokenizer

class ESGAnalyze():
  def __init__(self):
    self.model = AutoModel.from_pretrained(CONFIG['base_model_name'])
    self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model_name'])

  

def main():
  analyzer = ESGAnalyze()
  print(analyzer.model.config)

if __name__ == "__main__":
  main()