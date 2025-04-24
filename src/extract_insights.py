import json
from typing import Optional, TypedDict,Dict, Any
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from filter_articles import load_articles
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
def insights(text):
    class Features(BaseModel):
        Diseases: Optional[str] = Field(default=None, description="Relevant disease entities mentioned")
        Genes_Proteins: Optional[str] = Field(
            default=None, description="Gene and protein identifiers discussed"
        )
        Pathways: Optional[str] = Field(
            default=None, description="Biological pathways(A series of chemical reactions or interactions between molecules) referenced in the research"
        )
        Experimental_Methods: Optional[str] = Field(
            default=None, description="Techniques used like CRISPR or bulk RNASeq or single cell RNASeq etc"
        )

    class CombinedOutput(BaseModel):
        # Combines output of extracted features and summary
        structured_keywords: Optional[Features] = Field(description="Keywords and structured data extracted from the text")
        general_summary: Optional[str] = Field(description="A concise overall summary of the text")




    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )

    structured_llm = llm.with_structured_output(schema=Features)


    class GraphState(TypedDict):
        # State of Graph
        text: str
        extracted_features: Optional[Features] = None
        summary: Optional[str] = None



    def extract_features_node(state: GraphState) -> Dict[str, Any]:

        # Node to extract features from text
        print("--- Node: Extracting Features ---")
        llama_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "<<SYS>>\nYou are an expert information extraction algorithm. "
                    "Your job is to extract structured data from unstructured text based ONLY on the provided fields. "
                    "If an attribute cannot be found, respond with null for that field. "
                    "Return the output as a JSON object following the specified schema.\n<</SYS>>",
                ),
                (
                    "human",
                    "[INST]  Instruction:\n"
                    "Extract ONLY the following fields from the text:\n"
                    "- Diseases\n"
                    "- Genes_Proteins\n"
                    "- Pathways\n"
                    "- Experimental_Methods\n"
                    "If any field is missing, return null for that field.\n"
                    "Output must be a JSON object matching the schema.\n\n"
                    " Text:\n{text}\n\n"
                    "Output:\n [/INST]",
                ),
            ]
        )
        input_text = state['text']
        prompt = llama_prompt_template.invoke({"text": input_text})
        try:
            extracted_data = structured_llm.invoke(prompt)
            print("--- Extraction Successful ---")
            return {"extracted_features": extracted_data}
        except Exception as e:
            print(f"--- Feature Extraction Failed: {e} ---")
            return {"extracted_features": None}


    def summarize_text_node(state: GraphState) -> Dict[str, Any]:

        # Node for summary
        summary_prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert text summarizer.Just return the summary "
            ),
            (
                "human",
                "Please summarize the following text in 1-2 sentences revolving around the main scientific insight:\n\n{text}\n\nSummary:"

            )
        ])
        print("--- Node: Generating Summary ---")
        input_text = state['text']
        # print("****************")
        # print(input_text)
        # print("****************")

        prompt = summary_prompt_template.invoke({"text": input_text})
        # print("DEBUG: Type of object returned by invoke:", type(prompt))
        # print("****************")
        # print(prompt)
        # print("****************")
        try:
            response = llm.invoke(prompt)
            summary_text = response.content
            print("--- Summarization Successful ---")
            return {"summary": summary_text}
        except Exception as e:
            print(f"--- Summarization Failed: {e} ---")
            return {"summary": None}



    workflow = StateGraph(GraphState)


    workflow.add_node("extract_features", extract_features_node)
    workflow.add_node("summarize_text", summarize_text_node)
    #Flow
    workflow.set_entry_point("extract_features")
    workflow.add_edge("extract_features", "summarize_text")
    workflow.add_edge("summarize_text", END)

    # Compile the graph
    app = workflow.compile()
    initial_state = {"text": text}

    print("--- Invoking LangGraph Workflow ---")
    final_state = app.invoke(initial_state)
    print("--- LangGraph Workflow Complete ---")

    # Retrieve the results from the final state
    features = final_state.get("extracted_features")
    summary = final_state.get("summary")
    combined_output = CombinedOutput(
        structured_keywords=features,
        general_summary=summary
    )
    return combined_output.model_dump_json()



def transform_extraction_output(input_data):
  output_data = {}
  if isinstance(input_data.get("structured_keywords"), dict):
      output_data.update(input_data["structured_keywords"])

  summary = input_data.get("general_summary")
  if summary is not None:
      output_data["Key_Findings"] = summary
  elif "Key_Findings" not in output_data:
      output_data["Key_Findings"] = None

  return output_data

if __name__=="__main__":
    y = load_articles('../data/papers')
    with open("../outputs/filtered_articles.json", "r") as f:
        d = json.load(f)
    ext={}
    for i in d['filtered_articles']:
        insight=insights(y[i])
        insight = json.loads(insight)
        value=transform_extraction_output(insight)
        ext[i]=value
    with open("../outputs/extracted_insights.json", "w") as f:
        json.dump(ext, f, indent=2)

    print("JSON file saved to outputs/extracted_insights.json")

