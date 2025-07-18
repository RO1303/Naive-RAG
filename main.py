import sys
import bs4
import os
import requests
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import dotenv

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate

os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import IMSDbLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
client = QdrantClient(host="localhost", port=6333)


page_url = "https://imsdb.com/all-scripts.html"
r = requests.get(page_url)
if not r.status_code == 200:
    print(f"Problems with loading: {page_url}")
    sys.exit()

soup = bs4.BeautifulSoup(r.content, 'html.parser')

ms = soup.find("h1", string="All Movie Scripts on IMSDb (A-Z)")
ms = ms.parent.find_all("a", href=re.compile("^/Movie Scripts/"))
movie_scripts = {}
for script in ms:
    url = "https://imsdb.com/scripts/" + script.text.replace(" ", "-") + ".html"
    movie_scripts[script.text] = url

# for (num, key) in enumerate(movie_scripts.keys()):
#    print(f"{num+1}. {key}")

movie_title_input = input().strip()
movie_script_url = movie_scripts.get(movie_title_input, "NA")
if movie_script_url == "NA":
    print(f"Script for '{movie_title_input}'  wasn't found in the list of movie scripts.")
else:
    if requests.get(movie_script_url).status_code != 200:
        print(f"Script for '{movie_title_input}'  wasn't found in the list of movie scripts.")
    else:
        loader = IMSDbLoader(movie_script_url)
        print(f"Loaded script for {movie_title_input} from {movie_script_url}.")
        movie_text = loader.load()[0].page_content
        movie_text = re.sub(r'\s+', ' ', movie_text).strip()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=10,
            length_function=len,
            is_separator_regex=False,
            separators=["INT."]
        )
        texts = text_splitter.create_documents([movie_text])
        print(f"Found {len(texts)} scenes in the script for {movie_title_input}.")

        collection_name = movie_title_input.replace(" ", "-")
        if not client.collection_exists(collection_name=collection_name):
            vectors = embeddings.embed_documents([t.page_content for t in texts])
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )
            qdrant = QdrantVectorStore.from_documents(
                texts,
                embeddings,
                url="http://localhost:6333",
                prefer_grpc=True,
                collection_name=collection_name,
            )
        print(f"Embedded script for {movie_title_input}.")

        query = input()
        prompt = "Rewrite the query, ensuring that it emphasizes the necessary keywords for movie scene retrieval from a vector store. Query: " + query
        # prompt = "Write a passage that would be the perfect answer to the query: " + query
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | llm
        answer = chain.invoke({"question": query})
        print("Rewritten query to: ", answer.content)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        docs = vector_store.similarity_search(answer.content, k=1)
        print("Final Answer:")
        print("**INT. CAVE - DAY**")
        for (num, doc) in enumerate(docs):
            print(f"Scene {num+1}: {doc.page_content}")
