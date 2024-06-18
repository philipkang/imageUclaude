import streamlit as st
import boto3
import json
from PIL import Image
import io
import os

# Initialize Boto3 session and Bedrock client

#session = boto3.Session()
ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
SECRET_KEY = os.environ['AWS_SECRET_KEY']
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

bedrock = session.client(service_name='bedrock-runtime')

message_list = []

# Streamlit user interface for image upload
st.title("Image Description App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    # Display the uploaded image
    #st.image(image_bytes, caption="Uploaded Image", width=300)
    
    # Determine the image format
    image = Image.open(io.BytesIO(image_bytes))
    image_format = image.format.lower()
    if image_format == 'jpg':
        image_format = 'jpeg'  # Adjust 'jpg' to 'jpeg' for compatibility

    image_message = {
        "role": "user",
        "content": [
            { "text": "Image 1:" },
            {
                "image": {
                    "format": image_format,
                    "source": {
                        "bytes": image_bytes # no base64 encoding required!
                    }
                }
            },
            { "text": "Please describe the image." }
        ],
    }

    message_list.append(image_message)

    response = bedrock.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=message_list,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0
        },
    )

    response_message = response['output']['message']
    text_description = response_message['content'][0]['text']
    
    # Display the image and text response side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_bytes, caption="Uploaded Image", width=300)  # Adjust width as needed
    
    with col2:
        st.write(text_description)

    message_list.append(response_message)

    #st.write("Stop Reason:", response['stopReason'])
    #st.write("Usage:", json.dumps(response['usage'], indent=4))

    # User input for asking a question
    user_question = st.text_input("Ask a question about the image:")

    if user_question:
        question_message = {
            "role": "user",
            "content": [
                { "text": user_question }
            ]
        }

        message_list.append(question_message)

        response = bedrock.converse(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=message_list,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0
            },
        )

        response_message = response['output']['message']
        question_response = response_message['content'][0]['text']
        
        st.write(question_response)
        
        message_list.append(response_message)