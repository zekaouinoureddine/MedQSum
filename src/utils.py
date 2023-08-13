import evaluate


def get_rouge_scores(predictions, references):
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    return rouge_scores