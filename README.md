LangChain e Python: Integrando com Google Gemini ??

Este projeto demonstra como utilizar o LangChain para criar roteiros de viagem utilizando o modelo Gemini do Google.

?? Guia de Configuração

Siga os passos abaixo para configurar seu ambiente e utilizar os scripts do projeto.

1. Criar e Ativar Ambiente Virtual

Windows:

python -m venv langchain
langchain\Scripts\activate


Mac/Linux:

python3 -m venv langchain
source langchain/bin/activate


2. Instalar Dependências

Utilize o comando abaixo para instalar as bibliotecas necessárias (LangChain + Google Generative AI):

pip install -r requirements.txt


3. Configurar Chave de API

Crie um arquivo chamado .env na raiz do projeto (mesma pasta do main.py) e adicione sua chave do Google AI Studio:

GEMINI_API_KEY="SUA_CHAVE_DO_GOOGLE_AQUI"


Nota: Você pode obter sua chave gratuitamente no Google AI Studio.

4. Como Executar

O projeto já está configurado para ler automaticamente as variáveis do arquivo .env. Basta rodar:

python main.py
