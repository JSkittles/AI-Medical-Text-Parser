#Patient presents with symptoms consistent with scoliosis, including spinal curvature.
#Patient exhibits symptoms indicative of asthma

import tkinter as tk
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import re


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)


# Define a function to retrieve word definitions
def define_term(term):

    # Define the URL for the medical term.
    url = f'https://www.merriam-webster.com/dictionary/{term}#medicalDictionary'


    # Send an HTTP GET request to the URL.
    response = requests.get(url)


    # Check if the request was successful (status code 200).
    if response.status_code == 200:
        # Parse the HTML content of the page.
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the element that contains the definition.
        definition_element = soup.find('span', class_='dtText')



        if definition_element:
            # Extract the text of the definition.
            definition = definition_element.get_text()


            return definition.strip()
        else:
            definition = "{Definition not found.}"
            return definition
    else:
        definition = "{Was not able to find definition}"
        return definition




# Define a function to simplify and define text
def simplify_and_define(text, tokenWords):
    # Load the BERT model and tokenizer for sequence classification
    np.set_printoptions(suppress=True)
    model = BertForSequenceClassification.from_pretrained("CustomModel_v02.model")

    terms_to_define = []
    check_doubles = set()  # Use a set to keep track of terms and their definitions



    for word in tokenWords:
        # Tokenize the input text
        input = tokenizer(word, padding=True, truncation=True, return_tensors="pt")


        # Perform Prediction using the BERT model
        outputs = model(**input)


        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().detach().numpy()


        for predictOne in predictions:
            pred = predictOne[0]
            if pred < 0.7:
                terms_to_define.append(word)
            else:
                break


    # Clean the array list of terms
    terms_to_define = list(dict.fromkeys(terms_to_define))


    simplified_with_definitions = text


    for term in terms_to_define:
        definition = define_term(term)
        definition = definition.replace(":", "")
        if term not in check_doubles:
            simplified_with_definitions = simplified_with_definitions.replace(term, f"{term} {{ {definition}}} ")




            check_doubles.add(term)  # Add the term to the set
        else:
            check_doubles.add(term)  # Add the term to the set




    return simplified_with_definitions




# Create a function to handle the button click
def on_submit():
    user_input = entry.get("1.0", "end-1c")
   
    # Remove periods and non-letter characters using regular expressions
    cleaned_string = re.sub(r'[^a-zA-Z\s]', '', user_input)
    # Split the cleaned string into words
    word_list = cleaned_string.split()




    simplified_and_defined_text = simplify_and_define(user_input, word_list)
    #result_label.config(text=f"Simplified Text with Definitions:\n{simplified_and_defined_text}")




    entry2.config(state="normal")  # Enable the widget for editing
    entry2.delete("1.0", "end")  # Clear the existing content
    entry2.insert("1.0", simplified_and_defined_text)  # Insert the new text
    entry2.config(state="disabled")  # Disable the widget for editing




def close_gui():
    root.destroy()




#main
root = tk.Tk()
root.title('COMPLEX MEDICAL TEXT PARSER --------updated-10/19/23')
root.attributes('-fullscreen', True)






close_button = tk.Label(root, text="X", font=("Helvetica", 12), bg="red", fg="white")
close_button.pack(side="top", anchor="ne")
close_button.bind("<Button-1>", lambda e: close_gui())




# Create a Text widget for user input and a corresponding slider
entry = tk.Text(root, font=("Times New Roman", 13), wrap=tk.WORD)
entry.pack(fill=tk.BOTH, expand=False, padx=10, pady=4)
entry.insert("1.0", "ENTER TEXT [INPUT]. . .")  # Insert the new text




button = tk.Button(root, text="ANALYZE AND PARSE TEXT", command=on_submit)
button.pack(padx=10, pady=10)  # Adjust the pady as needed




# Create a second Text widget for user input and a corresponding slider
entry2 = tk.Text(root, font=("Times New Roman", 13), wrap=tk.WORD)
entry2.pack(fill=tk.BOTH, expand=False, padx=10, pady=6)
entry2.insert("1.0", "PARSED TEXT [OUTPUT]. . .")  # Insert the new text




result_label = tk.Label(root, text="")
result_label.pack()




root.mainloop()

