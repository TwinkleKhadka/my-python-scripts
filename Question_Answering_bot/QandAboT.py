import os
from crewai import Agent, Task, Crew, LLM
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

llm = LLM(model="huggingface/mistralai/Mistral-7B-Instruct-v0.3")

huggingfaceAPI="Your API Key"

#Loading document
def load_document(file):
    name, extension =os.path.splitext(file)
    if extension =='.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader=PyPDFLoader(file)
    elif extension== '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader=Docx2txtLoader(file)
    elif extension =='.txt':
        from langchain_community.document_loaders import TextLoader
        loader=TextLoader(file)
    else:
        print("Document not supported")
        return None
    data=loader.load()
    return data
#Chunking document
def chunk_data(data,chunk_size):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

#Creating embeddings from chunks
def create_embeddings(chunks):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    embeddings_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    vector_store=Chroma.from_documents(chunks,embeddings_model)
    return vector_store

#Using agent to answer quesstion as per retrieved text
def q_and_a(vector_store,k,q):
    from langchain_huggingface import HuggingFaceEndpoint
    from crewai.tools import tool
    import ast
    question = q
    def retrieve_relevant_text(vector_store, k,q):
        retriever_ = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
        docs = retriever_.invoke(q)
        return " ".join([doc.page_content for doc in docs])
    context = retrieve_relevant_text(vector_store,k, question)

    @tool("DeepSeek Answer Generator")
    def deepseek_tool(question: str, context: str = "") -> str:
        """Uses DeepSeek to answer questions from retrieved context."""
        deepseek = HuggingFaceEndpoint(repo_id="deepseek-ai/deepseek-chat", task="text-generation",
                                       model_kwargs={"max_length": 50},huggingfacehub_api_token=huggingfaceAPI)

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return deepseek.invoke(prompt)

    @tool("Ollama Answer Generator")
    def ollama_tool(question: str, context: str = "") -> str:
        """Uses Ollama to answer questions from retrieved context."""
        ollama = HuggingFaceEndpoint(repo_id="ollama-ai/ollama", task="text-generation",model_kwargs={"max_length": 50},huggingfacehub_api_token=huggingfaceAPI)
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return ollama.invoke(prompt)

    agent = Agent(role="AI Technology Researcher", goal="Retrieve separate answers from DeepSeek and Ollama within 50 words.",
                  backstory="A knowledgeable AI assistant that retrieves information using advanced models.",
                  tools=[deepseek_tool, ollama_tool],
                  llm=llm,
                  verbose=True, )
    # response = agent.run({"input": question, "context": context})
    task = Task(description=f"Answer the question: {question} using context: {context}, and by using two models.",
                expected_output="Two well-structured answers within 50 words based on the given context from two different llms in the from of dictionary as:{'Deepseek answer':answer1, 'Ollama answer':answer2}.Each answer should be within 50 words.",
                agent=agent)  # , input_dict={"question": question, "context": context})
    crew = Crew(agents=[agent], tasks=[task])
    response = crew.kickoff()
    #print('response', response)
    raw_response = response.tasks_output[0].raw if response.tasks_output else "{}"
    try:
        parsed_response = ast.literal_eval(raw_response)  # Convert string to dictionary
    except Exception as e:
        print("❌ Error parsing response:", e)
        parsed_response = {}
    Answer1 = parsed_response.get("Deepseek answer", "No DeepSeek response.")
    Answer2 = parsed_response.get("Ollama answer", "No Ollama response.")
    return (Answer1, Answer2)

#Comparing two answers from deepseek and ollama model
def compare_answer(question, answer1, answer2):
    from langchain.prompts import PromptTemplate
    from langchain_huggingface import HuggingFaceEndpoint
    Mistral = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation", huggingfacehub_api_token=huggingfaceAPI)
    evaluation_prompt = PromptTemplate(
        input_variables=['question', 'answer1', 'answer2'],
        template="""
            Question : {question}
            Deepseek :{answer1}
            Ollama :{answer2}
            Which answer is better and why? Please choose among two answers and provide very short reasoning for respective choice.
            """)
    evl_input = evaluation_prompt.format(question=question, answer1=answer1, answer2=answer2)
    print(evaluation_prompt)
    return Mistral.invoke(evl_input)

def clearhistory():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__=="__main__":
    st.subheader("LLM Question Answering Application")
    with st.sidebar:
        #Gemini_key=st.text_input('Gemini API key:', type='password')
        Huggingface_key=st.text_input('Hugging face API key:', type='password')
        os.environ["HUGGINGFACE_API_KEY"] = Huggingface_key
        uploaded_file=st.file_uploader('Upload a file:', type=['pdf','doc','txt'])
        chunk_size=st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clearhistory)
        k=st.number_input('k', min_value=1, max_value= 10, value=3, on_change=clearhistory)
        add_data=st.button("Add data", on_click=clearhistory)
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file'):
                bytes_data=uploaded_file.read()
                filename=os.path.join('./',uploaded_file.name)
                with open(filename,'wb') as f:
                    f.write(bytes_data)
                data=load_document(filename)
                chunks=chunk_data(data,chunk_size=chunk_size)
                st.write(f'Chunk sie : {chunk_size}, Chunks: {len(chunks)}')
                vector_store = create_embeddings(chunks)
                st.session_state.vs=vector_store
                st.success("Embedded successfully")
    q=st.text_input('Ask a question based on file provided:')
    answer=''
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k : {k}')
            Answer1, Answer2 = q_and_a(vector_store,k,q)
            st.text_area('Deepseek :',value=Answer1)
            st.text_area("Ollama:",value=Answer2)
            best_answer=compare_answer(Answer1,Answer2,q)
            st.text_area('Better answer: ', value=best_answer)

        st.divider()
        if 'history' not  in st.session_state:
            st.session_state.history =''
        value= f'Q: {q} \nA: {best_answer}'
        st.session_state.history= f'{value} \n {"*"*100} \n {st.session_state.history}'
        h= st.session_state.history
        st.text_area(label= "Chat history", value= h, key = 'history', height =500)

