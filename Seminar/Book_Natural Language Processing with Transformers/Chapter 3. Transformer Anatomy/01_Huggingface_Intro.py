from transformers import pipeline

question_answerer = pipeline('question-answering')
print(question_answerer({
        'question': 'What is the name of the repository ?',
        'context': 'Pipeline has been included in the huggingface/transformers repository'
    }))