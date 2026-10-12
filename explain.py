import os

from dotenv import load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_azure_ai.vectorstores.azuresearch import AzureSearch

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Load YouTube transcript
# ============================================================

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=8fEEbKJoNbU"
)

documents = loader.load()

print("YouTube transcript loaded.")


# ============================================================
# 3. Save transcript locally
# ============================================================

with open("youtube_transcript.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.page_content)
        f.write("\n\n")

print("Saved to youtube_transcript.txt")


# ============================================================
# 4. Split transcript into chunks
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")


# ============================================================
# 5. Save chunks locally
# ============================================================

os.makedirs("chunks", exist_ok=True)

for i, chunk in enumerate(chunks):

    file_path = f"chunks/chunk_{i + 1}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(chunk.page_content)

print("Chunks saved in the 'chunks' folder.")


# ============================================================
# 6. Azure OpenAI Embeddings
# ============================================================

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
)

print("Azure OpenAI embeddings initialized.")


# ============================================================
# 7. Azure AI Search configuration
# ============================================================

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX"]

SEMANTIC_CONFIG_NAME = "youtube-semantic-config"


# ============================================================
# 8. Connect to Azure AI Search Index Management
# ============================================================

index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=AzureKeyCredential(SEARCH_KEY),
)

print("Connected to Azure AI Search.")


# ============================================================
# Connect LangChain to Azure AI Search
# (creates the index if it does not exist yet)
# ============================================================

vector_store = AzureSearch(
    azure_search_endpoint=SEARCH_ENDPOINT,
    azure_search_key=SEARCH_KEY,
    index_name=INDEX_NAME,
    embedding_function=embeddings.embed_query,
)

print("LangChain Azure Search vector store initialized.")


# ============================================================
# 9. Get existing index
# ============================================================

existing_index = index_client.get_index(INDEX_NAME)

print(f"Found index: {INDEX_NAME}")


# ============================================================
# 10. Add semantic configuration
# ============================================================

semantic_config = SemanticConfiguration(
    name=SEMANTIC_CONFIG_NAME,
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[
            SemanticField(field_name="content")
        ]
    )
)


# Check whether semantic configuration already exists
semantic_search = existing_index.semantic_search

if semantic_search is None:

    existing_index.semantic_search = SemanticSearch(
        default_configuration_name=SEMANTIC_CONFIG_NAME,
        configurations=[
            semantic_config
        ],
    )

    print("Creating semantic configuration...")

else:

    existing_configs = semantic_search.configurations

    config_exists = any(
        config.name == SEMANTIC_CONFIG_NAME
        for config in existing_configs
    )

    if not config_exists:

        existing_configs.append(semantic_config)

        print("Adding semantic configuration...")

    else:

        print("Semantic configuration already exists.")

    existing_index.semantic_search.default_configuration_name = (
        SEMANTIC_CONFIG_NAME
    )


# ============================================================
# 11. Update Azure AI Search index
# ============================================================

index_client.create_or_update_index(existing_index)

print("Azure AI Search index updated.")
print(
    f"Semantic configuration: {SEMANTIC_CONFIG_NAME}"
)


# ============================================================
# 13. Upload chunks
# ============================================================

vector_store.add_documents(chunks)

print("Chunks successfully uploaded to Azure AI Search!")


# ============================================================
# 14. Search
# ============================================================

query = "What is the main topic discussed in the video?"

print()
print("Query:")
print(query)


# ============================================================
# 15. Vector similarity search
# (AzureSearch embeds the query itself; it does not support
#  similarity_search_by_vector)
# ============================================================

results = vector_store.similarity_search(
    query,
    k=4,
    search_type="similarity",
)


# ============================================================
# 17. Display results
# ============================================================

print()
print("================================================")
print("SEARCH RESULTS")
print("================================================")

for i, doc in enumerate(results, start=1):

    print()
    print(f"--------------- RESULT {i} ---------------")

    print(doc.page_content)

    print("--------------------------------------------")


print()
print("Done.")
