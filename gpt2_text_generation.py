from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    gen_kwargs = {
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 2,
        "do_sample": True,
    }
    
    with torch.no_grad():
        output_sequences = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

if __name__ == "__main__":
    print("GPT-2 Text Generation Model")
    print("Type 'exit' to quit.\n")
    
    while True:
        prompt = input("Enter your prompt: ")
        if prompt.strip().lower() == "exit":
            print("Exiting...")
            break
        output = generate_text(prompt, max_length=150)
        print("\nGenerated Text:\n", output)
        print("-" * 80)
