import os
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from .base_model import BaseRAGModel
import re

class GeminiRAGModel(BaseRAGModel):
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        configure(api_key=api_key)
        super().__init__(model_name)
        self._setup_social_responses()
    
    def _initialize_model(self):
        """Gemini model'ini başlat"""
        try:
            model_name = self.model_name or "gemini-1.5-flash"
            self.model = GenerativeModel(model_name)
        except Exception as e:
            print(f"Model {self.model_name} bulunamadı, varsayılan model deneniyor...")
            try:
                self.model = GenerativeModel("gemini-1.5-flash")
            except:
                self.model = GenerativeModel("gemini-pro")
    
    def _setup_social_responses(self):
        """Sosyal etkileşim cevaplarını ayarla"""
        self.social_responses = {
            'Greeting': [
                "Merhaba! Ben ilk yardım asistanınızım. Size nasıl yardımcı olabilirim?",
                "Günaydın! Sağlık konularında destek vermek için buradayım.",
                "Merhaba! Acil durumlar ve ilk yardım hakkında sorularınızı yanıtlayabilirim.",
                "Selamlar! Sağlığınızla ilgili endişeleriniz varsa yardımcı olmaya hazırım.",
                "İyi günler! İlk yardım konusunda sorularınızı yanıtlamaya hazırım."
            ],
            'Goodbye': [
                "İyi günler! Sağlığınıza dikkat edin ve gerekirse tekrar sorun.",
                "Hoşça kalın! Acil durumlarda yardım almaktan çekinmeyin.",
                "Görüşmek üzere! Sağlıklı günler dilerim.",
                "İyi günler! Sağlığınızla ilgili endişeleriniz olursa tekrar danışabilirsiniz.",
                "Hoşça kalın! Acil durumlarda tekrar sorabilirsiniz."
            ],
            'Thanks': [
                "Rica ederim! Sağlığınız için her zaman buradayım.",
                "Ne demek! Sağlık konularında yardımcı olmak benim görevim.",
                "Rica ederim! Acil durumlarda tekrar danışabilirsiniz.",
                "Çok rica ederim! Sağlığınızla ilgili endişeleriniz olursa yardımcı olmaya hazırım.",
                "Rica ederim! Başka sorularınız varsa yardımcı olmaya hazırım."
            ]
        }
        
        self.social_patterns = {
            'Greeting': [
                r'\b(merhaba|selam|selamlar)\b',
                r'\b(günaydın|iyi günler|iyi akşamlar|iyi geceler)\b',
                r'\b(hey|hi|hello|good morning|good afternoon|good evening)\b',
                r'\b(nasılsın|nasılsınız|naber|ne haber|ne var ne yok)\b',
                r'\b(hoş geldin|hoş geldiniz|welcome)\b'
            ],
            'Goodbye': [
                r'\b(hoşça kal|hoşça kalın|görüşürüz|görüşmek üzere)\b',
                r'\b(bye|goodbye|see you|take care)\b',
                r'\b(tamam|tamamdır|anladım|peki)\b'
            ],
            'Thanks': [
                r'\b(teşekkürler|teşekkür ederim|teşekkürler)\b',
                r'\b(thanks|thank you|thx)\b',
                r'\b(sağol|sağolun|sağolun var olun)\b',
                r'\b(çok teşekkürler|çok teşekkür ederim|çok sağolun)\b'
            ]
        }
    
    def is_social_interaction(self, query: str) -> str | None:
        query_lower = query.lower().strip()
        
        for intent, patterns in self.social_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return None
    
    def get_social_response(self, intent: str) -> str:
        import random
        if intent in self.social_responses:
            return random.choice(self.social_responses[intent])
        return ""
    
    def _create_qa_prompt(self, query: str, context: str) -> str:
        return f"""Aşağıdaki bilgilere dayanarak soruyu detaylı ve açıklayıcı şekilde yanıtla. Gerekiyorsa adım adım açıkla:

Bilgi: {context}

Soru: {query}

Cevap:"""
    
    def generate_answer(self, query: str, context: str) -> str:
        """Soru ve context'e göre cevap üret"""
        social_intent = self.is_social_interaction(query)
        if social_intent:
            return self.get_social_response(social_intent)
        
        try:
            prompt = self._create_qa_prompt(query, context)
            response = self.model.generate_content(prompt)
            
            if response.text:
                answer = response.text.strip()
                if len(answer) > 1000:
                    answer = answer[:1000] + "..."
                return answer
            else:
                return "Üzgünüm, bu konuda yeterli bilgi bulamadım."
                
        except Exception as e:
            print(f"Gemini model hatası: {e}")
            return "Üzgünüm, şu anda cevap üretemiyorum."

def test_model():
    """Model'i test et"""
    try:
        model = GeminiRAGModel()
        
        social_tests = [
            "Merhaba",
            "Selam!",
            "Günaydın",
            "İyi günler",
            "Hoşça kalın",
            "Teşekkürler",
            "Görüşürüz"
        ]
        
        print("=== Sosyal Etkileşim Testleri ===")
        for test in social_tests:
            response = model.generate_answer(test, "")
            print(f"Soru: {test}")
            print(f"Cevap: {response}")
            print("-" * 30)
        
        print("\n=== Normal RAG Testi ===")
        context = "Yanık durumunda soğuk su ile yıkayın."
        query = "Yanık durumunda ne yapmalıyım?"
        print("Cevap:", model.generate_answer(query, context))
        
    except ValueError as e:
        print(f"API Key hatası: {e}")
        print("GEMINI_API_KEY environment variable'ını ayarlayın.")
    except Exception as e:
        print(f"Model hatası: {e}")

if __name__ == "__main__":
    test_model() 