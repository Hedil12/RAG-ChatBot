import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from accelerate import Accelerator
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

accelerator = Accelerator()

# Define the RAG model and embeddings functions
def RAG_model(model_path="Qwen/Qwen2-0.5B-Instruct"):
    device = accelerator.device  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device=device,
        pad_token_id=0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def embeddings(modelPath="sentence-transformers/all-MiniLM-L12-v2"):
    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': accelerator.device}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    return HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

# Step 2: Retrieve Relevant Information
def retrieve_context(query, top_k=3):
    # Initialize Qdrant client with on-disk storage
    with Qdrant.from_existing_collection(path="Qdrant_VDB", collection_name="CS_doc", embedding=embeddings()) as vector_store:
        """Retrieve the most relevant documents for a given query."""
        docs = vector_store.similarity_search(query, k=top_k)
        context = " ".join([doc.page_content for doc in docs])
    return context


# Taking in the responses and models
def generate_response(model, tokenizer, text):
    model_inputs = tokenizer(text, return_tensors="pt").to("cpu")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id  # To avoid potential padding issues
    )
    
    return generated_ids

# Get question and answer (Main Function)
async def ask_question(user_query):
    prompt_template = """
    Your main role is to answer questions from the user. You are an assistant specializing in computer science principles and coding.
    Retrieve relevant information from the dataset and utilize inference and suggestions for the following tasks:
    - Responses should cover fundamental principles of computer science.
    - Inferences are allowed to provide comprehensive answers.
    - Use the provided context to list down relevant information and explanations.
    - Ensure all responses are accurate and aligned with computer science topics.
    Ensure responses are derived from the dataset, use inference and suggestions to provide comprehensive answers.
    """
    # Retrieve relevant context
    context = retrieve_context(user_query)
    model, tokenizer = RAG_model()

    # Prepare the prompt with context
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Context: {context}\n\n{user_query}"}
    ]

    # Concatenate the messages into a single string for the model
    text = "\n".join([f"{message['role']}: {message['content']}" for message in messages])

    # Tokenize and generate response
    generated_ids = generate_response(model, tokenizer, text)

    # Decode the generated response
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract the response after the user query
    cleaned_response = clean_response(generated_text)

    return cleaned_response

# Clean the responses
def clean_response(generated_text):
    response_start = generated_text.find("Answer:")
    if response_start != -1:
        cleaned_response = generated_text[response_start + len("Answer:"):].strip()
    else:
        cleaned_response = generated_text.strip()

    cleaned_response = "\n\n".join([line.strip() for line in cleaned_response.split("\n\n") if line.strip()])
    return cleaned_response

def main(query):
    return ask_question(query)

if __name__ == '__main__':
    query = input('Enter query: ')
    main(query)
    