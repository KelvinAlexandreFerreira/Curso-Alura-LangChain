from langchain_core.prompts import PromptTemplate
from gemini_setup import get_gemini_llm
from langchain.globals import set_debug
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import ChatPromptTemplate

llm = get_gemini_llm()
set_debug(True)

class Destino(BaseModel):
    cidade: str = Field(description="cidade a visitar")
    motivo: str = Field(description="motivo pelo qual a cidade é interessante")

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = PromptTemplate(
    template="""Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()},
)

modelo_restaurante = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

parte1 = modelo_cidade | llm | parseador
parte2 = modelo_restaurante | llm | StrOutputParser()
parte3 = modelo_cultural | llm | StrOutputParser()

cadeia = (parte1 | parte2 | parte3)

try:
    resultado = cadeia.invoke({"interesse": "praias"})
    print("\n--- Resultado Final ---")
    print(resultado)
except Exception as e:
    print(f"Ocorreu um erro: {e}")