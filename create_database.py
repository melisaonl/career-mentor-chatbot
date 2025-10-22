from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import pandas as pd

# Load the CSV Dataset
csv_path = "data/career_chatbot_dataset.csv"
df = pd.read_csv(csv_path)

# Format the data into a single string
def make_content(row):
    return "\n".join([
        f"Job Title: {row.get('job_title','')}",
        f"Description: {row.get('job_description','')}",
        f"Skills: {row.get('job_skill_set','')}",
        f"Category: {row.get('category','')}",
        f"Field of Study: {row.get('field_of_study','')}"
    ])

# Convert rows to LangChain Document objects
docs = [
    Document(
        page_content=make_content(r),
        metadata={
            "job_title": r.get("job_title",""),
            "category": r.get("category",""),
            "source": "career_chatbot_dataset.csv"
        }
    )
    for _, r in df.iterrows()
]

print(f"Total {len(docs)} documents are ready.")
print(docs[0].page_content[:300])

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split text into chunks
texts = text_splitter.split_documents(docs)

# Initialize Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and Persist Chroma Vector Database
db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./career_chroma_db"
)
db.persist()
print("Vektör veritabanı başarıyla oluşturuldu ve kaydedildi: ./career_chroma_db")
