__author__= "Amin Amiri"
__version__ = "v05312024"
#T5 tester


from transformers import T5ForConditionalGeneration, T5Tokenizer

# load the fine-tuned model and tokenizer from the local directory
model = T5ForConditionalGeneration.from_pretrained('./t5-title-generation/checkpoint-500/')
tokenizer = T5Tokenizer.from_pretrained('./t5-title-generation/checkpoint-500/')

# Amin: or load the model and tokenizer from Hugging Face Hub
# model = T5ForConditionalGeneration.from_pretrained('your-username/t5-title-generation')
# tokenizer = T5Tokenizer.from_pretrained('your-username/t5-title-generation')

#process input func
def preprocess_input(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    return inputs

# example text input
new_text = "Machine learning is increasingly moving from hand-designed models to automatically optimized pipelines using tools such as H20, TPOT, and auto-sklearn. These libraries, along with methods such as random search, aim to simplify the model selection and tuning parts of machine learning by finding the best model for a dataset with little to no manual intervention. However, feature engineering, an arguably more valuable aspect of the machine learning pipeline, remains almost entirely a human labor."
inputs = preprocess_input(new_text)

# generate the title
outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)

# decode the output
generated_title = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Title:", generated_title)

print("\n\nThe program finished successfully.\n")
