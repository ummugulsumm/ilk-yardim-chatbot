from transformers.pipelines import pipeline
from .base_model import BaseRAGModel
import re
from transformers import AutoTokenizer

class FlanT5RAGModel(BaseRAGModel):
    def __init__(self, model_name="google/flan-t5-base"):
        super().__init__(model_name)
        self._setup_social_responses()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _initialize_model(self):
        """Flan-T5 model'ini başlat"""
        self.qa_pipe = pipeline(
            "text2text-generation", 
            model=self.model_name, 
            max_length=384, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5
        )
    
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
    
    def is_social_interaction(self, query):
        """Sorgunun sosyal etkileşim olup olmadığını kontrol et"""
        query_lower = query.lower().strip()
        
        for intent, patterns in self.social_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return None
    
    def get_social_response(self, intent):
        """Sosyal etkileşim için cevap döndür"""
        import random
        if intent in self.social_responses:
            return random.choice(self.social_responses[intent])
        return ""
    
    def _create_qa_prompt(self, query, context):
        """Cevap üretimi için daha açıklayıcı ve detaylı cevap isteyen prompt oluştur"""
        return f"""Aşağıdaki bilgilere dayanarak soruyu Türkçe olarak detaylı ve açıklayıcı şekilde yanıtla. Gerekiyorsa adım adım açıkla:

Bilgi: {context}

Soru: {query}

Cevap:"""
    
    def _remove_surrogates(self, text):
        """Surrogate karakterleri temizle"""
        return re.sub(r'[\ud800-\udfff]', '', text)
    
    def _remove_repetitions(self, text):
        """Art arda tekrar eden cümle veya kelime gruplarını temizle"""
        return re.sub(r'(\b\w+(?:\s+\w+){0,5}\b)(?:\s*\1\b)+', r'\1', text)
    
    def _truncate_context(self, context, query, max_input_tokens=510):
        """Prompt+context toplam token sayısını 510'u aşmayacak şekilde context'i kademeli olarak kısalt"""
        base_prompt = f"Aşağıdaki bilgilere dayanarak soruyu Türkçe olarak detaylı ve açıklayıcı şekilde yanıtla. Gerekiyorsa adım adım açıkla:\n\nBilgi: "
        suffix = f"\n\nSoru: {query}\n\nCevap:"
        context_words = context.split()
        while True:
            joined_context = ' '.join(context_words)
            prompt = base_prompt + joined_context + suffix
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) <= max_input_tokens or len(context_words) <= 10:
                return joined_context
            context_words = context_words[:-max(1, int(len(context_words)*0.05))]
    
    def generate_answer(self, query, context):
        """Soru ve context'e göre cevap üret"""
        social_intent = self.is_social_interaction(query)
        if social_intent:
            return self.get_social_response(social_intent)
        context = self._truncate_context(context, query, max_input_tokens=510)
        try:
            prompt = self._create_qa_prompt(query, context)
            result = self.qa_pipe(prompt, max_length=384, do_sample=True, temperature=0.7)
            answer = result[0]['generated_text'].strip()
            answer = self._remove_surrogates(answer)
            answer = self._remove_repetitions(answer)
            if answer and len(answer) < 10:
                return "Üzgünüm, bu konuda yeterli bilgi bulamadım."
            if answer and len(answer) > 1000:
                answer = answer[:1000] + "..."
            return answer
        except Exception as e:
            print(f"Model hatası: {e}")
            return "Üzgünüm, şu anda cevap üretemiyorum."

def test_model():
    """Model'i test et"""
    model = FlanT5RAGModel()
    
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

if __name__ == "__main__":
    test_model() 