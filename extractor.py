from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import PromptTemplate, LLMChain
import torch
from ocr import extract_text_from_image

'''
Make sure to set hugging face token as environment variable:
    export HUGGINGFACE_TOKEN=your_hugging_face_token
'''
def extract_rules_for_making_machine(doc_content):
    # Load the Mistral 8B model and tokenizer from Hugging Face Hub
    model_name = "mistralai/Mistral-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    # Define the prompt template
    prompt_template = PromptTemplate(
        # input_variables=["document"],
        template=doc_content
    )

    # Create the LLMChain
    llm_chain = LLMChain(
        llm=model,
        prompt_template=prompt_template,
        tokenizer=tokenizer
    )

    # Generate the prompt
    prompt = prompt_template.format(document=doc_content)

    # Generate the response
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=512)
    rules = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the rules from the response
    rules_start = rules.find("Rules:") + len("Rules:")
    extracted_rules = rules[rules_start:].strip()

    return extracted_rules

# Example usage
file_path = 'path/to/your/image/file.png'
header_height_cm = 2.0 # Height of the header in cm
footer_height_cm = 2.0 # Height of the footer in cm
doc_content = extract_text_from_image(file_path, header_height_cm, footer_height_cm)
# print(document_text)

# Example usage
prompt = f"""
You are an expert in manufacturing processes and mechanical engineering. Please read the provided content carefully and extract the key rules, steps, or processes involved in machine making. Focus on understanding the sequence of operations, materials, tools, and techniques that are necessary for creating machines.

Document:
{doc_content}
"""
rules = extract_rules_for_making_machine(prompt)
print(rules)
