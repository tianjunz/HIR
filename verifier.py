import torch
import torch.nn.functional as F


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def tracking_shuffled_objects_three_objects(prefix, response, target):
    prompt = ""
    neg_prompt = ""
    correct = 0
    final_answer_context = response
    if response.replace(" ", "").replace(".", "").lower() == str(
        target
    ).lower().replace("\n", " ").replace(" ", "").replace(".", ""):
        # if str(target).lower().replace('\n', ' ').replace(' ', '').replace('.', '') in response.replace(' ', '').replace('.', '').lower() and len(response.replace(' ', '').replace('.', '').lower()) - len(str(target).lower().replace('\n', ' ').replace(' ', '').replace('.', '')) < 4:
        prompt = prompt + " Solve question with correct answer."
        neg_prompt = neg_prompt + " Solve the question with wrong answer."
        correct = 1
    else:
        if "Bob" in response or "Alice" in response or "Claire" in response:
            # prompt = prompt + " Solve the question with only people names."
            prompt = prompt + " Solve the question with wrong answer."
            neg_prompt = neg_prompt + " Solve the question with correct answer."
        else:
            prompt = prompt + " Solve the question with wrong answer."
            neg_prompt = neg_prompt + " Solve the question with correct answer."
    # print(response, target, correct)

    return prompt, neg_prompt, correct
