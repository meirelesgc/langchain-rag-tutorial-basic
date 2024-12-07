from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

# Carrega as variáveis de ambiente. Assume que o projeto contém um arquivo .env com as chaves da API.
load_dotenv()
# ---- Define a chave da API da OpenAI
# Altere o nome da variável de ambiente de "OPENAI_API_KEY" para o nome especificado no
# seu arquivo .env.
openai.api_key = os.environ["OPENAI_API_KEY"]


def main():
    first_sentence = input("Primeira palavra: ")
    second_sentence = input("Primeira palavra: ")

    # Obtém o embedding de uma palavra.
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query(first_sentence)
    # print(f"Vetor para 'apple': {vector}")
    print(f"Tamanho do vetor: {len(vector)}")

    # Compara os vetores de duas palavras.
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = (first_sentence, second_sentence)
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparando ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
