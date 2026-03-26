# 🔎 Day 11: Hallucination Detector (Gradio + LangChain)

An AI-powered tool to identify and highlight potential hallucinations in any text. It extracts factual claims and verifies them using high-speed Groq models.

## 🚀 Features
- **Gradio Interface**: A clean, modern UI for easy interaction.
- **Claim Extraction**: Automatically identifies verifiable factual claims within a text.
- **Multi-Step Verification**: Uses LangChain to verify each claim individually for higher accuracy.
- **Detailed Reporting**: Provides a hallucination score and a comprehensive final report.
- **Premium Design**: Dark mode with violet/indigo accents.

## 🛠️ Built With
- **Gradio**: For the frontend.
- **LangChain**: For prompt orchestration and claim-by-claim analysis.
- **Groq AI (Llama 3 70B)**: For logical verification.

## 🏃 How to Run
1. Navigate to the project directory:
   ```bash
   cd "day11_hallucination_detector"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Enter your **Groq API Key** in the UI.

---
*Built as part of the 25 Days of ML Challenge.*
