import os
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega a chave do .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERRO: Chave GEMINI_API_KEY não encontrada no arquivo .env")
else:
    print(f"✅ Chave encontrada: {api_key[:5]}...{api_key[-3:]}")
    
    # Configura o acesso
    genai.configure(api_key=api_key)

    print("\n🔍 Consultando o Google sobre modelos disponíveis...")
    
    try:
        # Lista os modelos que suportam geração de texto
        found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
                found = True
        
        if not found:
            print("⚠️ Conexão feita, mas nenhum modelo de texto foi retornado. Verifique permissões da chave.")
            
    except Exception as e:
        print(f"\n❌ Erro de conexão: {e}")