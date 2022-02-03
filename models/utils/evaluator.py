import torch
import numpy as np
from tqdm import tqdm
import torch


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True  # False

    model.eval()

    return model


class Evaluator:

    def evaluate(self, img, data_generator, model):

        start_token_id = data_generator.tokenizer.encode(data_generator.sos_token)[0]

        tokens = []
        tokens.append(start_token_id)

        softmaps_list = []

        result = []

        for i in range(0, data_generator.max_length):
            input_sequence = torch.Tensor(tokens).unsqueeze(0).long()  # .cuda()
            logits, softmaps = model(input_sequence, img)
            softmap = softmaps[:, i, :]
            softmaps_list.append(softmap.view(softmap.shape[-1]))

            _, predicted_indices = torch.max(logits[:, i, :], dim=1)
            predicted_token_id = int(predicted_indices[0])
            predicted_token = data_generator.tokenizer.decode(predicted_token_id)

            result.append(predicted_token)
            tokens.append(predicted_token_id)

            if predicted_token_id == data_generator.tokenizer.eos_token_id:
                break
        final_sentence = ' '.join(res for res in result)
        print(final_sentence)
        return final_sentence, softmaps_list

    def beam_search(self, data_generator, model, img, beam=3):
        start_token_id = data_generator.tokenizer.encode(data_generator.sos_token)[0]

        tokens = [start_token_id]

        result = []

        for i in range(0, data_generator.max_length):

            beam_scores = [1.0] * len(data_generator.tokenizer)  # one score for each input
            tokens_list = [[] for _ in range(len(data_generator.tokenizer))]

            for b in range(beam):

                for word_idx in tqdm(range(len(data_generator.tokenizer))):
                    current_tokens = tokens
                    current_tokens.extend(tokens_list[word_idx])
                    beam_input = torch.Tensor(current_tokens).unsqueeze(0).long()  # .cuda()
                    logits, _ = model(beam_input, img)

                    scores, predicted_indeces = torch.max(logits[:, i + b, :], dim=1)  # get max likely next word
                    beam_scores[word_idx] *= int(scores[0])  # get likelihood score
                    tokens_list[word_idx].append(int(predicted_indeces[0]))

            predicted_token_id = np.argmax(beam_scores)
            predicted_token = data_generator.tokenizer.decode(predicted_token_id)

            result.append(predicted_token)
            tokens.append(predicted_token_id)

            if predicted_token_id == data_generator.tokenizer.eos_token_id:
                break

        final_sentence = ' '.join(res for res in result)
        return final_sentence
