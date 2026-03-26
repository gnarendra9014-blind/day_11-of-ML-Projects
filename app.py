import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# LangChain Chains
# ─────────────────────────────────────────────

def build_chains(api_key: str):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

    # Chain 1: Extract claims from the text
    claim_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Extract all factual claims from the following text.
A factual claim is a specific, verifiable statement (names, dates, numbers, events, scientific facts, etc.)

Text: {text}

Return ONLY a JSON array of claims. No preamble, no explanation.
Example: ["Claim 1", "Claim 2", "Claim 3"]

Claims:"""
    )

    # Chain 2: Verify each claim using Groq's knowledge
    verify_prompt = PromptTemplate(
        input_variables=["claim"],
        template="""You are a rigorous fact-checker. Evaluate this claim based on your knowledge:

Claim: {claim}

Respond ONLY in this exact JSON format:
{{
  "verdict": "TRUE" | "FALSE" | "UNCERTAIN",
  "confidence": <number 0-100>,
  "explanation": "<one sentence explanation>",
  "correction": "<corrected fact if FALSE, else null>"
}}

JSON:"""
    )

    # Chain 3: Overall hallucination report
    report_prompt = PromptTemplate(
        input_variables=["text", "verification_results"],
        template="""You are an expert at detecting AI hallucinations. Given the original text and the claim verification results, produce a final hallucination assessment.

Original Text:
{text}

Claim Verification Results:
{verification_results}

Write a concise hallucination report with:
1. Overall Hallucination Risk: LOW / MEDIUM / HIGH
2. Summary of findings (2-3 sentences)
3. Most problematic claims (if any)
4. Recommendation for the reader

Keep it professional and under 150 words."""
    )

    claim_chain = claim_prompt | llm
    verify_chain = verify_prompt | llm
    report_chain = report_prompt | llm

    return claim_chain, verify_chain, report_chain


def detect_hallucinations(text: str, api_key: str):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key.", "", ""
    if not text.strip():
        return "⚠️ Please enter some text to analyze.", "", ""
    if len(text.strip()) < 30:
        return "⚠️ Text too short. Please enter at least a sentence or two.", "", ""

    try:
        claim_chain, verify_chain, report_chain = build_chains(api_key)

        # Step 1: Extract claims
        claims_response = claim_chain.invoke({"text": text})
        raw_claims = claims_response.content.strip()

        # Parse JSON claims
        try:
            json_match = re.search(r'\[.*?\]', raw_claims, re.DOTALL)
            claims = json.loads(json_match.group()) if json_match else []
        except Exception:
            claims = []

        if not claims:
            return "ℹ️ No verifiable factual claims found in the text.", "", ""

        # Step 2: Verify each claim
        results = []
        claims_display = ""

        for i, claim in enumerate(claims[:8], 1):  # max 8 claims
            try:
                raw_result_response = verify_chain.invoke({"claim": claim})
                raw_result = raw_result_response.content.strip()
                json_match = re.search(r'\{.*?\}', raw_result, re.DOTALL)
                result = json.loads(json_match.group()) if json_match else {}

                verdict = result.get("verdict", "UNCERTAIN")
                confidence = result.get("confidence", 50)
                explanation = result.get("explanation", "Could not verify.")
                correction = result.get("correction")

                # Emoji for verdict
                emoji = {"TRUE": "✅", "FALSE": "❌", "UNCERTAIN": "⚠️"}.get(verdict, "⚠️")

                claim_block = f"""**{emoji} Claim {i}:** {claim}
- **Verdict:** {verdict} (Confidence: {confidence}%)
- **Explanation:** {explanation}"""
                if correction:
                    claim_block += f"\n- **Correction:** {correction}"

                claims_display += claim_block + "\n\n---\n\n"
                results.append({"claim": claim, "verdict": verdict, "confidence": confidence, "explanation": explanation})

            except Exception as e:
                claims_display += f"**⚠️ Claim {i}:** {claim}\n- Could not verify: {str(e)}\n\n---\n\n"
                results.append({"claim": claim, "verdict": "UNCERTAIN", "confidence": 0, "explanation": "Verification failed."})

        # Step 3: Overall report
        report_response = report_chain.invoke({
            "text": text,
            "verification_results": json.dumps(results, indent=2)
        })
        report = report_response.content.strip()

        # Step 4: Score summary
        true_count = sum(1 for r in results if r["verdict"] == "TRUE")
        false_count = sum(1 for r in results if r["verdict"] == "FALSE")
        uncertain_count = sum(1 for r in results if r["verdict"] == "UNCERTAIN")
        total = len(results)

        score_display = f"""### 📊 Hallucination Score

| | Count | % |
|---|---|---|
| ✅ True | {true_count} | {true_count*100//total}% |
| ❌ False | {false_count} | {false_count*100//total}% |
| ⚠️ Uncertain | {uncertain_count} | {uncertain_count*100//total}% |
| 📋 Total Claims | {total} | — |

**Hallucination Rate: {false_count*100//total}%**
"""

        return score_display, claims_display, report

    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

EXAMPLES = [
    ["Albert Einstein was born in Germany in 1879 and won the Nobel Prize in Physics in 1922 for his discovery of the law of photoelectric effect. He later moved to the United States in 1933 and worked at Princeton University until his death in 1955."],
    ["The Eiffel Tower was built in 1887 and stands 450 meters tall. It was designed by Gustave Eiffel as a permanent structure for the 1889 World's Fair in Paris. Over 10 million people visit it every year."],
    ["Python was created by Guido van Rossum and first released in 1991. It is currently the most popular programming language in the world according to the TIOBE index. Python 4.0 was released in 2023."],
]

css = """
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    max-width: 1100px !important;
    margin: auto !important;
}
.header-box {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid #7c3aed;
}
.header-box h1 { color: white; font-size: 2rem; margin: 0; }
.header-box p { color: #a78bfa; margin-top: 0.5rem; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="Day 11 – Hallucination Detector") as demo:

    gr.HTML("""
    <div class="header-box">
        <div style="display:inline-block;background:#7c3aed;color:white;padding:4px 14px;border-radius:20px;font-size:0.75rem;font-weight:700;letter-spacing:1px;margin-bottom:0.75rem;">DAY 11 · 25-DAY AI CHALLENGE</div>
        <h1>🔍 Hallucination Detector</h1>
        <p>Paste any AI-generated or human-written text · Extract claims · Verify with Groq · Get a hallucination report</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="🔑 Groq API Key",
                placeholder="gsk_...",
                type="password",
                value=os.getenv("GROQ_API_KEY", "")
            )
            input_text = gr.Textbox(
                label="📝 Text to Analyze",
                placeholder="Paste any text here — AI output, article, Wikipedia paragraph, etc.",
                lines=10
            )
            analyze_btn = gr.Button("🔍 Detect Hallucinations", variant="primary", size="lg")

            gr.Examples(
                examples=EXAMPLES,
                inputs=input_text,
                label="💡 Try an Example"
            )

        with gr.Column(scale=1):
            score_out = gr.Markdown(label="📊 Score")
            with gr.Accordion("📋 Claim-by-Claim Breakdown", open=True):
                claims_out = gr.Markdown(label="Claims")
            with gr.Accordion("📝 Final Hallucination Report", open=True):
                report_out = gr.Markdown(label="Report")

    gr.HTML("""
    <div style='text-align:center;margin-top:1rem;font-size:0.8rem;color:#6b7280;'>
    Stack: Python · LangChain · Groq · LLaMA 3 70B · Gradio &nbsp;|&nbsp; Cost: $0.00
    </div>
    """)

    analyze_btn.click(
        fn=detect_hallucinations,
        inputs=[input_text, api_key],
        outputs=[score_out, claims_out, report_out]
    )

if __name__ == "__main__":
    demo.launch()