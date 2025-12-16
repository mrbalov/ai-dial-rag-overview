import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


"""
export DIAL_API_KEY="<SECRET>" &&
py -m venv .venv &&
source .venv/bin/activate &&
pip install -r requirements.txt &&
python3 -m task
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        # Store provided clients and prepare vectorstore (loads or creates index)
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system.

        If an index folder exists locally we load the FAISS index from disk to avoid
        re-creating embeddings every run. Otherwise we create a new index from the
        microwave manual text file and save it locally for future reuse.
        """
        print("🔄 Initializing Microwave Manual RAG System...")

        index_folder = "microwave_faiss_index"

        # If the local index exists, load it to save time and API calls
        if os.path.exists(index_folder) and os.path.isdir(index_folder):
            print(f"Found existing index at '{index_folder}', loading it...")
            # NOTE: allow_dangerous_deserialization may be required depending on
            # the langchain-community FAISS implementation. It enables loading
            # pickled files that might be unsafe in untrusted environments.
            # This is acceptable for local development and tests but avoid in PROD.
            try:
                vect = FAISS.load_local(folder_path=index_folder, embeddings=self.embeddings,
                                        allow_dangerous_deserialization=True)
                print("✅ Loaded FAISS index from disk.")
                return vect
            except TypeError:
                # Some versions may use a different kwarg name; try without it.
                vect = FAISS.load_local(folder_path=index_folder, embeddings=self.embeddings)
                print("✅ Loaded FAISS index from disk (fallback call).")
                return vect

        # Otherwise create a new index from the documents
        print("No local index found — creating a new one...")
        return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        # Load the raw text file containing the microwave manual
        # Use an absolute path relative to this file to ensure the loader can find
        # the manual regardless of the current working directory when the script
        # is executed.
        base_dir = os.path.dirname(__file__)
        manual_path = os.path.join(base_dir, "microwave_manual.txt")

        loader = TextLoader(file_path=manual_path, encoding="utf-8")
        documents = loader.load()

        # Split the loaded documents into smaller chunks so embeddings capture local context
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50,
                                                  separators=["\n\n", "\n", "."])
        chunks = splitter.split_documents(documents)

        # Create FAISS vectorstore from the document chunks using provided embeddings
        vect = FAISS.from_documents(chunks, self.embeddings)

        # Persist index locally for faster startup next time
        vect.save_local("microwave_faiss_index")
        print("💾 Saved new FAISS index to 'microwave_faiss_index'.")

        return vect

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        # Print a visual separator and the step header for clarity
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # Perform similarity search returning both documents and their scores.
        # Some vectorstore implementations accept a score threshold kwarg which may
        # have different semantics (similarity vs distance). To avoid accidentally
        # filtering all results we call the method without that kwarg and apply our
        # own filtering logic below.
        try:
            results = self.vectorstore.similarity_search_with_score(query=query, k=k)
        except TypeError:
            # If the implementation signature differs, try the older similarity_search
            # without scores and fall back to returning entire documents.
            try:
                docs = self.vectorstore.similarity_search(query=query, k=k)
                results = [(d, None) for d in docs]
            except Exception as e:
                print(f"Error during similarity search: {e}")
                results = []

        context_parts = []

        # results may be a list of (Document, score) tuples or just Documents.
        # We will filter manually based on the provided `score` threshold.
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, scr = item
            else:
                doc, scr = item, None

            # Print raw score for debugging
            if scr is not None:
                print(f"- raw score: {scr}")

            # Determine whether to keep this document based on score semantics.
            # Previous logic attempted to treat scores <=1 as similarities and >1 as distances
            # and could discard all top-k results when the vectorstore returned distances
            # with values larger than the threshold. To be robust across implementations
            # we will accept the top-k results by default and only apply filtering when
            # the score clearly represents a similarity in the [0,1] range.

            keep = True
            if scr is not None:
                try:
                    scr_val = float(scr)
                    if 0.0 <= scr_val <= 1.0:
                        # treat as similarity (higher is better)
                        keep = scr_val >= score
                    else:
                        # For scores outside [0,1] we assume distance semantics or an
                        # alternate scaling. Instead of discarding, keep the top-k results
                        # and avoid applying a possibly incompatible threshold.
                        keep = True
                except Exception:
                    # If we cannot interpret the score, keep the doc to be safe
                    keep = True

            # Print a preview of the page content to help debugging
            print(f"- content preview: {getattr(doc, 'page_content', str(doc))[:200]}...")

            if not keep:
                print("- discarded by score threshold")
                continue

            # Add full content of the doc chunk to context parts
            content = getattr(doc, 'page_content', None)
            if content:
                context_parts.append(content)

        if not context_parts:
            print("⚠️ No relevant context retrieved. The RAG CONTEXT will be empty.")

        print("=" * 100)
        # Join selected chunks using double newlines for readability
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        # Inject retrieved context and user query into the USER_PROMPT template
        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        # Prepare messages expected by the chat LLM: system instructions + human prompt
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]

        # Call the LLM client to generate a response. Different langchain versions
        # return different payload shapes, so we try to handle a few.
        # NOTE: `BaseChatModel.generate` expects a list of message-lists (batched).
        # Passing `messages=[messages]` wraps the single conversation into a batch of 1.
        response = self.llm_client.generate(messages=[messages])

        # Extract text from response with fallbacks for different response shapes
        content = None
        try:
            # Some versions: response.generations is a list of lists of generation objects
            content = response.generations[0][0].text
        except Exception:
            try:
                # Another possible shape: generation objects have a message with content
                content = response.generations[0][0].message.content
            except Exception:
                # Fallback to stringifying the whole response
                content = str(response)

        print(f"Assistant: {content}")
        return content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        # If the user types exit commands, break the loop
        if user_question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # Step 1: Retrieval of context
        context = rag.retrieve_context(user_question)
        # Step 2: Augmentation - combine retrieved context and user question
        augmented = rag.augment_prompt(user_question, context)
        # Step 3: Generation - ask the LLM to produce an answer
        answer = rag.generate_answer(augmented)

        # Print final assistant answer (already printed in generate_answer but keep for clarity)
        print(f"\nFinal answer:\n{answer}\n")

if __name__ == "__main__":
    main(
        MicrowaveRAG(
            # Create embeddings and LLM client using Azure endpoints and provided API key
            embeddings=AzureOpenAIEmbeddings(deployment="text-embedding-3-small-1",
                                             azure_endpoint=DIAL_URL,
                                             api_key=SecretStr(API_KEY)),
            llm_client=AzureChatOpenAI(
                                    # temperature=0.0,
                                       azure_deployment="gpt-5-mini-2025-08-07",
                                       azure_endpoint=DIAL_URL,
                                       api_key=SecretStr(API_KEY),
                                       api_version="2024-12-01-preview")
        )
)
