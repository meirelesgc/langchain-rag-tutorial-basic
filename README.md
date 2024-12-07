# Tutorial Langchain RAG

## Instale as dependências

1. Realize os seguintes passos antes de instalar as dependências listadas no arquivo `requirements.txt`, devido aos desafios atuais para instalar o `onnxruntime` via `pip install onnxruntime`.

    - Para usuários de **MacOS**, uma solução alternativa é instalar primeiro a dependência `onnxruntime` necessária para o `chromadb` utilizando:

    ```python
    conda install onnxruntime -c conda-forge
    ```
    Consulte este [tópico](https://github.com/microsoft/onnxruntime/issues/11037) para obter ajuda adicional, se necessário.

    - Para usuários de **Windows**, siga o guia [aqui](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) para instalar o Microsoft C++ Build Tools. Certifique-se de concluir todas as etapas, incluindo a configuração da variável de ambiente no caminho.

2. Execute este comando para instalar as dependências do arquivo `requirements.txt`:

```python
pip install -r requirements.txt
```

3. Instale as dependências para markdown com:

```python
pip install "unstructured[md]"
```

## Crie o banco de dados

Crie o banco de dados Chroma DB:

```python
python create_database.py
```

## Consulte o banco de dados

Realize consultas no Chroma DB:

```python
python query_data.py "Como Alice conhece o Chapeleiro Maluco?"
```

> Você também precisará criar uma conta na OpenAI (e configurar a chave da OpenAI na variável de ambiente) para que isso funcione.

Aqui está um tutorial em vídeo, passo a passo: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).