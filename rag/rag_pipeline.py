import os
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
import re

class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) pipeline sınıfı"""
    
    def __init__(self, pdf_path, chroma_dir="chroma_db"):
        self.pdf_path = pdf_path
        self.chroma_dir = chroma_dir
        self.client = None
        self.collection = None
        self._initialize_database()
        self._load_pdf()
    
    def _initialize_database(self):
        """ChromaDB veritabanını başlat"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name="ilk_yardim_knowledge",
                metadata={"description": "İlk yardım bilgileri"}
            )
            
        except Exception as e:
            print(f"Veritabanı başlatma hatası: {e}")
            raise
    
    def _load_pdf(self):
        """PDF dosyasını yükle ve işle"""
        if not os.path.exists(self.pdf_path):
            print(f"PDF dosyası bulunamadı: {self.pdf_path}")
            return
        
        try:
            reader = PdfReader(self.pdf_path)
            text_chunks = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    chunks = self._split_text(text, chunk_size=500, overlap=50)
                    text_chunks.extend(chunks)
            
            if self.collection.count() == 0:
                print(f"PDF'den {len(text_chunks)} parça çıkarıldı, veritabanına ekleniyor...")
                
                for i, chunk in enumerate(text_chunks):
                    if chunk.strip():
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[{"source": f"page_{i//3}", "chunk_id": i}],
                            ids=[f"chunk_{i}"]
                        )
                
                print("✅ PDF veritabanına başarıyla eklendi!")
            else:
                print("✅ Veritabanı zaten mevcut, PDF yükleme atlanıyor.")
                
        except Exception as e:
            print(f"PDF yükleme hatası: {e}")
            raise
    
    def _split_text(self, text, chunk_size=500, overlap=50):
        """Metni belirtilen boyutta parçalara böl"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def get_context(self, query, k=5):
        """Sorguya en uygun context'i getir"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            if results['documents'] and results['documents'][0]:
                context = ' '.join(results['documents'][0])
                return context
            else:
                return "İlgili bilgi bulunamadı."
                
        except Exception as e:
            print(f"Context alma hatası: {e}")
            return "Veritabanı hatası."

def get_rag_pipeline(pdf_path, chroma_dir="chroma_db"):
    return RAGPipeline(pdf_path, chroma_dir)

def get_context(query, pdf_path, chroma_dir="chroma_db", k=5):
    pipeline = get_rag_pipeline(pdf_path, chroma_dir)
    return pipeline.get_context(query, k) 