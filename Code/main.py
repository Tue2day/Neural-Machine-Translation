import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from datapreprocessing import input_lang, output_lang, train_data, val_data, test_pairs
from model import Encoder, Decoder, LanguageModelCriterion, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train(model, data, start_prob, end_prob, num_epochs=100):
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.

        # Calculate the probability of using the target sequence (Teacher Forcing)
        TF = True
        teacher_forcing_prob = start_prob - (start_prob - end_prob) * (epoch / num_epochs)
        random_prob = random.random()
        if random_prob > teacher_forcing_prob:
            TF = False

        for iteration, (b_x, b_x_len, b_y, b_y_len) in enumerate(data):
            b_x = torch.from_numpy(b_x).to(device).long()
            b_x_len = torch.from_numpy(b_x_len).to(device).long()

            dec_b_input = torch.from_numpy(b_y[:, :-1]).to(device).long()  # Before EOS
            # In training mode, decoder's input contains no "EOS"
            dec_b_output = torch.from_numpy(b_y[:, 1:]).to(device).long()  # After BOS
            # In training mode, standard decoder's output contains no "BOS"

            b_y_len = torch.from_numpy(b_y_len - 1).to(device).long()
            b_y_len[b_y_len <= 0] = 1

            if TF:
                # Teacher Forcing
                b_pred, attn = model(b_x, b_x_len, dec_b_input, b_y_len)
            else:
                bos = torch.Tensor([[output_lang.word2index["BOS"]]]).long().to(device)
                # Free Running
                b_pred, attn = model.free_running(b_x, b_x_len, bos, dec_b_output.size(1))

            # Generate mask between prediction and standard output
            b_out_mask = torch.arange(b_y_len.max().item(), device=device)[None, :] < b_y_len[:, None]
            b_out_mask = b_out_mask.float()

            # Calculate cross-entropy
            loss = loss_fn(b_pred, dec_b_output, b_out_mask)

            num_words = torch.sum(b_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if iteration % 100 == 0:
                print("Epoch: ", epoch, 'iteration ', iteration, 'loss: ', loss.item())

        avg_training_loss = total_loss / total_num_words
        training_losses.append(avg_training_loss)
        print("Epoch: ", epoch, "Training loss: ", avg_training_loss)

        if epoch % 10 == 0:
            val_loss = evaluate(model, val_data)
            validation_losses.append(val_loss)
        else:
            validation_losses.append(None)  # Placeholder for plotting

    # torch.save(model.state_dict(), '../Checkpoint/translate_model_TF.pt')
    torch.save(model.state_dict(), '../Checkpoint/translate_model_FR.pt')

    # Plotting the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot([i for i in range(num_epochs) if i % 10 == 0],
             [x for x in validation_losses if x is not None], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.show()


def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.

    with torch.no_grad():
        for _, (b_x, b_x_len, b_y, b_y_len) in enumerate(data):
            b_x = torch.from_numpy(b_x).to(device).long()
            b_x_len = torch.from_numpy(b_x_len).to(device).long()

            dec_b_input = torch.from_numpy(b_y[:, :-1]).to(device).long()
            dec_b_output = torch.from_numpy(b_y[:, 1:]).to(device).long()

            b_y_len = torch.from_numpy(b_y_len - 1).to(device).long()
            b_y_len[b_y_len <= 0] = 1

            b_pred, attn = model(b_x, b_x_len, dec_b_input, b_y_len)

            b_out_mask = torch.arange(b_y_len.max().item(), device=device)[None, :] < b_y_len[:, None]
            b_out_mask = b_out_mask.float()

            loss = loss_fn(b_pred, dec_b_output, b_out_mask)

            num_words = torch.sum(b_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

    avg_val_loss = total_loss / total_num_words
    print("Validation loss: ", avg_val_loss)
    return avg_val_loss


def translate(index, beam_width):
    original_input_list = [input_lang.index2word[i] for i in test_pairs[index][0]]
    original_input = " ".join(original_input_list)  # Original input
    standard_output = " ".join([output_lang.index2word[i] for i in test_pairs[index][1]])  # Standard output

    # Get one sentence to translate
    x = torch.from_numpy(np.array(test_pairs[index][0]).reshape(1, -1)).long().to(device)
    # x.shape = (1, seq_len)
    x_len = torch.from_numpy(np.array([len(test_pairs[index][0])])).long().to(device)
    # x_len.shape = (1, 1)
    bos = torch.Tensor([[output_lang.word2index["BOS"]]]).long().to(device)
    # bos.shape = (1, 1)
    eos = torch.Tensor([[output_lang.word2index["EOS"]]]).long().to(device)
    # eos.shape = (1, 1)

    translation, attn_weights = model.translate(x, x_len, bos, eos, beam_width)
    # In test mode, decoder's input is "BOS"

    translation = [output_lang.index2word[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break

    # Print original input and standard output
    print(original_input)
    print(standard_output)
    print(" ".join(trans))

    return trans, standard_output.split()[1:-1], original_input_list[1:-1], np.array(attn_weights)


# Calculate BLEU score
def calculate_bleu_scores(test_nums, beam_width):
    total_bleu_score = 0
    smoothing_function = SmoothingFunction().method1

    for i in range(test_nums):
        translated_sentence, reference_sentence, _, _ = translate(i, beam_width)
        bleu_score = sentence_bleu([reference_sentence], translated_sentence, smoothing_function = smoothing_function)
        total_bleu_score += bleu_score

    average_bleu_score = total_bleu_score / test_nums
    print(f"Average BLEU score: {average_bleu_score:.4f}")


# Attention Visualization
def plot_attention(attn_weights, input_sequence, output_sequence):
    # Set font to support Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # Ensure that minus sign is displayed correctly

    # Plotting parameters
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap="YlGnBu", xticklabels=input_sequence, yticklabels=output_sequence, cbar=True, annot=True, fmt=".2f")
    plt.xlabel('Input sequence')
    plt.ylabel('Output sequence')
    plt.title('Attention Heatmap')
    plt.show()


if __name__ == "__main__":
    dropout = 0.2
    embed_size = hidden_size = 512
    num_layers = 2
    beam_width = 3
    start_prob = 1.0
    end_prob = 1.0

    encoder = Encoder(vocab_size=input_lang.total_words,
                      embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                      dec_hidden_size=hidden_size,
                      num_layers=num_layers,
                      dropout=dropout)
    decoder = Decoder(vocab_size=output_lang.total_words,
                      embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                      dec_hidden_size=hidden_size,
                      num_layers=num_layers,
                      dropout=dropout)

    model = Seq2Seq(encoder, decoder)
    model = model.to(device)
    loss_fn = LanguageModelCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Train model
    # start_time = time.time()
    # train(model, train_data, start_prob, end_prob, num_epochs=50)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print()
    # # print("Free Running with Beam Search:")
    # print("Teacher Forcing with Beam Search:")
    # print(f'The total training time of the model is {elapsed_time:.2f} s')

    # Import the trained model
    # model.load_state_dict(torch.load('../Checkpoint/translate_model_TF.pt', map_location=device))
    model.load_state_dict(torch.load('../Checkpoint/translate_model_FR.pt', map_location=device))

    # test_nums = len(test_pairs)
    # calculate_bleu_scores(test_nums, beam_width)

    # Test Example
    for i in range(278, 292):
        tran, _, original_input_list, attn_weights = translate(i, beam_width)
        plot_attention(attn_weights[:-1, 1:-1], original_input_list, tran)
        print()



