import chromadb
import requests
import uuid
from datetime import datetime

def get_embedding(text):

    res = requests.client.post("https://local:11434/api/embeddings", jason{

        "model" = "nomic-embed-text",
        "propmt" = "text"
    })
    res.raise_get_status()
    return res.json()["embedding"]
  
  except Exception as e :
    return none

client = chromadb.client(
    chromadb.config.setting("persist_directory" = "./chromadb")
)

collection = client.get_or_create_collection("memory")

def add_memory(text, id = none, metadata = none):

    emd = get_embedding(text)
    if emd is none:
        return none

    mem_id = id or str(uuid.uuid4())

    meta = metadata or {}
    meta["created_at"] = datetime.utcnow().isoform()

    collection.add(

        id = [mem_id],
        metadata = [meta],
        documents = [text],
        embedding = [emd]
    )

    client.persist()
    return mem_id

def retrieve(querry, k = 3, filter = none):

    emd = get_embedding(querry)
    if emd is none:
        return []

    result = collection.querry(

        querry_embedding = [emd],
        n_results = k,
        where = filter,
    )

    output = []

    docs = result.get("documents", [[]])[0]
    meta = result.get("metadata", [[]])[0]
    dist = result.get("distance", [[]])[0]

    for d, m, dist in zip(documents, metadata, distance):
        output.append({

            "documents" = d,
            "metadata" = m,
            "distance" = dist,

        })

        return output
    
