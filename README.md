# PDF-reader
Talk to your PDF using Google Gemini!

## Document Ingestion and Preprocessing
- I used LangChain’s document loaders to ingest data from various formats (PDF, DOCX, etc.). I then chunked the content using
  RecursiveCharacterTextSplitter, ensuring each chunk had enough context without being too large for the model.

## Vector Store Setup
- Embedded the document chunks using GoogleGenerativeAIEmbeddings. These embeddings are stored in a vector FAISS database for fast similarity search.

## Retriever Construction
- Used LangChain’s VectorStoreRetriever to fetch the most relevant document chunks based on the user's query. This ensures that the model always gets contextually relevant information.

## Prompt Engineering & QA Chain
- Created a custom prompt template to guide the LLM to generate responses using the retrieved context. LangChain’s RetrievalQA chain tied everything together—retrieving context and generating accurate answers using an LLM like gemini.

## Evaluation and UX
- Built a Streamlit frontend for user interaction.
