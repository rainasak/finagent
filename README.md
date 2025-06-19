# finagent
Autonomous Financial Agent

# finagent
Autonomous Financial Agent

## Local Setup Instructions

Follow these steps to set up and run the project locally on your machine.

### 1. Clone the Repository

```powershell
# Windows PowerShell
# Or use your preferred terminal

git clone <your-repo-url>
cd financial-agent

# Windows
python -m venv venv
.\venv\Script\activate

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate

# Install Requirements
pip install -r requirements.txt

# Intall the package locally
pip install -e .

# Set Up Environment Variables
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
# Add any other required keys here

# Run the App
streamlit run main.py