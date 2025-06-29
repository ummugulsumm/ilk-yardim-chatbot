from abc import ABC, abstractmethod

class BaseRAGModel(ABC):
    """RAG modelleri için temel sınıf"""
    
    def __init__(self, model_name=None):
        self.model_name = model_name
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Model'i başlat - alt sınıflar implement etmeli"""
        pass
    
    @abstractmethod
    def generate_answer(self, query, context):
        """Soru ve context'e göre cevap üret"""
        pass
    
    def _create_qa_prompt(self, query, context):
        """Cevap üretimi için standart prompt oluştur"""
        return f"Soru: {query}\nCevap (sadece aşağıdaki bilgilere dayanarak):\n{context}\nCevap:" 