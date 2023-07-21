import logging
import os
import shutil
import subprocess

from flask import Flask, jsonify, request
from waitress import serve

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma

from werkzeug.utils import secure_filename

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY


model_id = os.getenv("MODEL_ID", default="TheBloke/orca_mini_3B-GGML")
model_basename = os.getenv("MODEL_BASENAME", default="orca-mini-3b.ggmlv3.q4_0.bin")
temp = os.getenv("TEMP", default=0)
n_ctx = os.getenv("N_CTX", default=2048)
n_threads = os.getenv("N_THREADS", default=4)
PROMPT_PATH = "prompt.txt"
prompt_template = ""

DEVICE_TYPE = "cpu"
SHOW_SOURCES = True
EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})


logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

if os.path.exists(PROMPT_PATH):
    try:
        f = open(PROMPT_PATH, "r")
        prompt_template = f.read()
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")
else:
    print("The prompt path does not exist")

if os.path.exists(PERSIST_DIRECTORY):
    try:
        shutil.rmtree(PERSIST_DIRECTORY)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")
else:
    print("The directory does not exist")

run_langest_commands = ["python3", "ingest.py"]
if DEVICE_TYPE == "cpu":
    run_langest_commands.append("--device_type")
    run_langest_commands.append(DEVICE_TYPE)

result = subprocess.run(run_langest_commands, capture_output=True)
if result.returncode != 0:
    raise FileNotFoundError(
        "No files were found inside SOURCE_DOCUMENTS, please put a starter file inside before starting the API!"
    )

# load the vectorstore
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

RETRIEVER = DB.as_retriever()


# load the LLM for generating Natural Language responses
def load_model(model_id, model_basename, temp, n_ctx, n_threads):

    logging.info(f"Loading Model: {model_id}")

    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)

    return LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        max_tokens=n_ctx,
        temperature=temp,
        repeat_penalty=1.15,
        n_threads=n_threads
    )


LLM = load_model(model_id, model_basename, temp, n_ctx, n_threads)
QA = RetrievalQA.from_chain_type(
    llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
)

app = Flask(__name__)


@app.route("/api/delete_source", methods=["GET"])
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"

    print("Deleting source documents")

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})


@app.route("/api/save_document", methods=["GET", "POST"])
def save_document_route():

    print("Adding new document")

    if "document" not in request.files:
        return "No document part", 400

    file = request.files["document"]
    if file.filename == "":
        return "No selected file", 400

    if file:
        filename = secure_filename(file.filename)
        folder_path = "SOURCE_DOCUMENTS"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        file.save(file_path)
        return "File saved successfully", 200


@app.route("/api/run_ingest", methods=["GET"])
def run_ingest_route():
    global DB
    global RETRIEVER
    global QA
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")

        run_langest_commands = ["python3", "ingest.py"]
        if DEVICE_TYPE == "cpu":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(DEVICE_TYPE)

        print("Running ingest")
        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        # load the vectorstore
        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()

        QA = RetrievalQA.from_chain_type(
            llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
        )
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    global QA
    global prompt_template

    target = request.json["target"]
    ram = request.json["RAM"]
    so = request.json["SO"]
    cloud_providers = request.json["cloud_providers"]
    num = request.json["num"]

    user_prompt = prompt_template.replace("<TARGET>", target).replace(
        "<RAM>", ram).replace("<SO>", so).replace("<CLOUD_PROVIDERS>", cloud_providers).replace("<NUM>", num)

    print("Processing prompt")
    res = QA(user_prompt)
    answer, docs = res["result"], res["source_documents"]
    print(answer)

    prompt_response_dict = {
        "Prompt": user_prompt,
        "Answer": answer,
    }

    prompt_response_dict["Sources"] = []
    for document in docs:
        prompt_response_dict["Sources"].append(
            (os.path.basename(str(document.metadata["source"])), str(document.page_content))
        )

    return jsonify(prompt_response_dict), 200


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    # app.run(debug=False, port=5110)
    serve(app, port=5110)
