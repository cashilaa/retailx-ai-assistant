## Project Overview

RetailX AI Assistant is an intelligent system designed to analyze retail data and answer user queries about customers, products, and sales. This project leverages the power of LangChain, LangGraph, and the Llama 3 model to create an agentic AI workflow that can autonomously handle data retrieval tasks and generate human-readable responses.

## Features

- Automated SQL query generation based on natural language questions
- Intelligent decision-making on whether a question can be answered with available data
- Execution of SQL queries on a SQLite database
- Generation of human-readable responses to user queries
- User-friendly interface powered by Streamlit

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Llama API key
- LangSmith API key (for LangChain tracing)

## Installation

1. Clone the repository:
git clone https://github.com/cashilaa/retailx-ai-assistant
cd retailx-ai-assistant
Copy
2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Copy
3. Install the required packages:
pip install -r requirements.txt
Copy
4. Set up your API keys:
- Open `main.py`
- Replace `"YOUR_LLAMA_API_KEY"` with your actual Llama API key
- Replace `"YOUR_LANGSMITH_API_KEY"` with your actual LangSmith API key

## Usage

1. Start the Streamlit app:
streamlit run streamlit_app.py
Copy
2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your question in the text input field and click "Submit".

4. The AI assistant will process your question and provide an answer based on the available retail data.

## Project Structure

- `main.py`: Contains the core logic, including data preparation, workflow definition, and AI model integration.
- `streamlit_app.py`: Defines the user interface using Streamlit.
- `requirements.txt`: Lists all the Python dependencies required for the project.
- `retail.db`: SQLite database containing the sample retail data.

## Customization

- To modify the sample data, edit the `data` dictionary in `main.py`.
- To add new functionalities or change the workflow, modify the relevant sections in `main.py`.
- To customize the user interface, edit `streamlit_app.py`.

## Troubleshooting

- If you encounter any import errors, ensure that all dependencies are correctly installed and that you're using a compatible Python version.
- For API-related issues, double-check that your API keys are correctly set in `main.py`.
- If you experience unexpected behavior, check the console output for error messages and logging information.

## Contributing

Contributions to improve RetailX AI Assistant are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- LangChain and LangGraph for providing powerful tools for building AI applications
- Anthropic for the Llama 3 model
- The Streamlit team for their excellent web app framework

Project Link: https://github.com/cashilaa/retailx-ai-assistant
