import torch
import numpy as np
from rouge_score import rouge_scorer
from torch.nn.functional import cosine_similarity

class MetricCalculator:
    def __init__(self, model, word_to_index, index_to_word, X_test, y_test, device):
        self.model = model
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def calculate_perplexity(self, test_loader):
        self.model.eval()
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
        return np.exp(total_loss / len(test_loader))

    def calculate_cosine_sim(self, test_loader):
        self.model.eval()
        total_sim = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                target_embs = self.model.embeddings(targets)
                pred_embs = outputs @ self.model.embeddings.weight
                total_sim += cosine_similarity(pred_embs, target_embs).mean().item()
                
        return total_sim / len(test_loader)

    def calculate_rouge(self, test_samples=100):
        self.model.eval()
        rouge_scores = []
        
        for i in np.random.choice(len(self.X_test), test_samples, replace=False):
            context = self.X_test[i].unsqueeze(0).to(self.device)
            reference = self.index_to_word[self.y_test[i].item()]
            
            generated_idxs = self.model.generate(context)
            generated_text = ' '.join([self.index_to_word[idx] for idx in generated_idxs])
            
            scores = self.scorer.score(reference, generated_text)
            rouge_scores.append(scores['rougeL'].fmeasure)
            
        return np.mean(rouge_scores)