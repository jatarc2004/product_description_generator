import hashlib
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import gradio as gr
import json
import inspect
import os

# Azure Computer Vision credentials
COMPUTER_VISION_ENDPOINT = "https://ai-aihackthonhub282549186415.cognitiveservices.azure.com/"
COMPUTER_VISION_API_KEY = "Fj1KPt7grC6bAkNja7daZUstpP8wZTXsV6Zjr2FOxkO7wsBQ5SzQJQQJ99BCACHYHv6XJ3w3AAAAACOGL3Xg"
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://ai-aihackthonhub282549186415.openai.azure.com")
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "Fj1KPt7grC6bAkNja7daZUstpP8wZTXsV6Zjr2FOxkO7wsBQ5SzQJQQJ99BCACHYHv6XJ3w3AAAAACOGL3Xg")

# Initialize the ComputerVision client
credentials = CognitiveServicesCredentials(COMPUTER_VISION_API_KEY)
vision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, credentials)

# In-memory user database for demo purposes
USERS_FILE = "users.json"
USERS_HISTORY_FILE = "users_history.json"

siz=0
current_user="!!"

if os.path.exists(USERS_FILE):
    try:
        with open(USERS_FILE, "r") as f:
            users_db = json.load(f)
    except json.decoder.JSONDecodeError:
        users_db = {}
else:
    users_db = {}

if os.path.exists(USERS_HISTORY_FILE):
    try:
        with open(USERS_HISTORY_FILE, "r", encoding="utf-8") as f:
            users_history_db = json.load(f)
    except json.decoder.JSONDecodeError:
        users_history_db = {}
else:
    users_history_db = {}

def save_history():
    with open(USERS_HISTORY_FILE, "w") as f:
        json.dump(users_history_db, f, indent=4)

def save_users():
    """Persist the users_db dictionary to the JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users_db, f, indent=4)

prompt_file = "prompt_template.txt"

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def show_history(username):
    siz = len(users_history_db[username])
    lis = []
    for i in range(siz):
        lis.append(users_history_db[username][i])
    cleaned = inspect.cleandoc("\n".join(lis))  # ✅ Convert list to a single string
    return cleaned



# Login function: verifies credentials and shows/hides the main UI
def login(username, password):
    hashed_input = hash_password(password)
    if username in users_db and users_db[username] == hashed_input:
        return "Login successful!", gr.update(visible=True)
    else:
        return "Invalid credentials. Please try again.", gr.update(visible=False)

# Signup function: registers a new user if the username is not already taken
def signup(username, password):
    if username in users_db:
        return "Username already exists. Please choose another username."
    else:
        users_db[username] = hash_password(password)
        save_users()
        return "Signup successful! Please log in with your new credentials."

# Function to extract keywords from an image using Azure Computer Vision
def extract_keywords_from_image(image_file):
    with open(image_file, "rb") as image:
        # Analyze the image for tags
        analysis = vision_client.analyze_image_in_stream(image, visual_features=[VisualFeatureTypes.tags])
    # Extract the tags (keywords) from the analysis result
    keywords = [tag.name for tag in analysis.tags]
    return ", ".join(keywords)  # Return as comma-separated string

# Function to generate product description with optional user-provided keywords

class ProductDescGen(LLMChain):
    """LLM Chain for generating product descriptions with emojis."""

    @classmethod
    def from_llm(cls, llm, prompt, **kwargs):
        return cls(llm=llm, prompt=prompt, **kwargs)

def product_desc_generator_with_image(username, product_name, image_file, user_keywords, target_language, word_limit):
    # Extract keywords from the image
    image_keywords = extract_keywords_from_image(image_file)
    
    # Combine the user-provided keywords with the image keywords
    combined_keywords = user_keywords + ", " + image_keywords if user_keywords else image_keywords

    # Assumes that 'prompt_file', 'PromptTemplate', 'AzureChatOpenAI', and 'ProductDescGen'
    # are defined elsewhere. For demonstration, this returns a placeholder string.
    with open(prompt_file, "r") as file:
        prompt_template = file.read()

    PROMPT = PromptTemplate(
        input_variables=["product_name", "keywords", "target_language", "word_limit"], template=prompt_template
    )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-05-01-preview",
        temperature=0.7
    )

    ProductDescGen_chain = ProductDescGen.from_llm(llm=llm, prompt=PROMPT)
    siz = 0
    if(username in users_history_db):
        siz=len(users_history_db[username])
    ProductDescGen_query = ProductDescGen_chain.run(
        product_name=product_name, keywords=combined_keywords, target_language=target_language, word_limit=word_limit
    )
    ProductDescGen_query += "\n"
    ProductDescGen_query = f"# PRODUCT {siz+1} ( Product Name - {product_name}, Target Language - {target_language}, Word Limit - {word_limit} )- \n" + ProductDescGen_query
    if username not in users_history_db:
        users_history_db[username] = []
    users_history_db[username].append(ProductDescGen_query)
    save_history()
    # print(ProductDescGen_query)
    return ProductDescGen_query

# Gradio UI
with gr.Blocks() as demo:
    # Login/Signup Section
    with gr.Tab("Login/Signup"):
        with gr.Tabs():
            # Login Tab
            with gr.Tab("Login"):
                username_login = gr.Textbox(label="Username", placeholder="Enter username")
                password_login = gr.Textbox(label="Password", placeholder="Enter password", type="password")
                login_status = gr.Textbox(label="Status", interactive=False)
                login_button = gr.Button("Login")
            # Signup Tab
            with gr.Tab("Signup"):
                username_signup = gr.Textbox(label="Username", placeholder="Choose a username")
                password_signup = gr.Textbox(label="Password", placeholder="Choose a password", type="password")
                signup_status = gr.Textbox(label="Status", interactive=False)
                signup_button = gr.Button("Signup")
    
    # Main Product Description Generator (hidden until login is successful)
    main_ui = gr.Column(visible=False)
    with main_ui:
        gr.HTML("<h1>Welcome to Product Description Generator</h1>")
        gr.Markdown(
            "Generate SEO-friendly product descriptions instantly! Provide a product name, an image, and optionally enter product keywords."
        )
        with gr.Tabs():
            with gr.Tab("Generate Product Description from Image!"):
                product_name = gr.Textbox(label="Product Name", placeholder="Nike Shoes")
                image_file = gr.Image(type="filepath", label="Upload Image")
                print(image_file.value)
                user_keywords = gr.Textbox(label="Enter Keywords (Optional)", placeholder="black shoes, leather shoes")
                target_language = gr.Dropdown(
                    choices=["English", "Spanish", "French", "German", "Italian","Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu",
                        "Gujarati", "Malayalam", "Kannada", "Punjabi", "Odia", "Assamese", "Maithili"],
                    label="Select Output Language",
                    value="English"  # Default language
                )
                word_limit = gr.Dropdown(
                    choices=[50, 100, 150, 200, 250, 300],  # Word limits options
                    label="Select Word Limit",
                    value=100  # Default word limit
                )
                product_description = gr.Markdown()
                click_button = gr.Button(value="Generate Description!")
                click_button.click(
                    product_desc_generator_with_image, 
                    [username_login, product_name, image_file, user_keywords, target_language, word_limit], 
                    product_description
                )
            with gr.Tab("User History"):
                history_button = gr.Button("Display History!")
                history_user = gr.Markdown()


    # Set up interactions for login and signup buttons
    history_button.click(
        show_history,
        username_login,
        history_user
    )

    login_button.click(
        login, 
        [username_login, password_login], 
        [login_status, main_ui]
    )
    
    signup_button.click(
        signup, 
        [username_signup, password_signup], 
        signup_status
    )

demo.launch(share=True)
