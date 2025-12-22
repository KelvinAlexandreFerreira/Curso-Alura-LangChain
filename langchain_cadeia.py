from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from gemini_setup import get_gemini_llm
from langchain.globals import set_debug

modelo_cidade = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {input}. A sua saída deve ser SOMENTE o nome da cidade. Cidade: "
)

modelo_restaurante = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}. "
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}. "
)

llm = get_gemini_llm()
set_debug(True)

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_restaurantes = LLMChain(prompt=modelo_restaurante, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(
    chains=[cadeia_cidade, cadeia_restaurantes, cadeia_cultural],
    verbose=True
)

resultado = cadeia.invoke({"input": "praias"})
print(resultado)