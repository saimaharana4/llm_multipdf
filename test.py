# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from sentence_transformers import SentenceTransformer

# class CustomHuggingFaceInstructEmbeddings(HuggingFaceInstructEmbeddings):
#     def __init__(self, model_name: str):
#         self.client = SentenceTransformer(model_name)
#         self.model_name = model_name

# def get_vectorstore(text_chunks):
#     try:
#         embeddings = CustomHuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#         vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#         return vectorstore
#     except Exception as e:
#         raise e

# # Test the function
# text_chunks = ["This is a test chunk."]
# try:
#     vectorstore = get_vectorstore(text_chunks)
#     print("Vectorstore created successfully.")
# except Exception as e:
#     print(f"Error: {e}")


# try:
#     from huggingface_hub.snapshot_download import REPO_ID_SEPARATOR
#     print("huggingface_hub module is correctly installed.")
# except ImportError as e:
#     print(f"Error importing huggingface_hub: {e}")
# import langchain_community
# print(dir(langchain_community.chains))