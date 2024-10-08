# Course Recommendation Chatbot 🐝

This is an AI-powered course recommendation chatbot that provides personalized course suggestions based on user interests and goals. This Streamlit application uses advanced natural language processing and retrieval techniques to offer tailored educational recommendations.

## Features

- Interactive chat interface
- Personalized course recommendations
- Integration with OpenAI's GPT models
- Efficient retrieval using FAISS vectorstore
- Conversation memory for context-aware responses

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/course-recommendation-chatbot.git
   cd course-recommendation-chatbot
   ```

2. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run course_recommendation_chatbot.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Start chatting with the bot and get personalized course recommendations!

## How It Works

1. The chatbot uses a FAISS index to efficiently retrieve relevant course information based on user queries.
2. It employs the LangChain library to create a conversational retrieval chain, combining retrieved information with a language model.
3. The chatbot maintains conversation history to provide context-aware responses.
4. Responses are generated using a custom prompt template that ensures detailed and personalized course recommendations.

## Customization

- Modify the `PROMPT_TEMPLATE` in the script to adjust the chatbot's behavior and response style.
- Update the `REQUIRED_PACKAGES` list if you need to add or remove dependencies.
- Adjust the language model settings in the `setup_qa_chain()` function to use different models or parameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the LangChain library for building the conversational AI pipeline.
- Course data is retrieved using a FAISS index for efficient similarity search.

---

For any questions or support, please open an issue in the GitHub repository.
