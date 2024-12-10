import torch
import torch.nn as nn
import torch.nn.functional as F


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input.shape = (batch_size, seq_len, vocab_size)
        # target.shape = mask.shape = (batch_size, seq_len)
        input = input.contiguous().view(-1, input.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)

        # Calculate cross-entropy(decoder use log_softmax as classified unction)
        output = -input.gather(1, target) * mask
        # why mask? Eliminate the effect of target's padding element "0"

        output = torch.sum(output) / torch.sum(mask)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.num_layers = num_layers
        self.enc_hidden_size = enc_hidden_size

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        # hid.shape = (4, batch_size, enc_hidden_size)

        # Process all layers
        hid = hid.view(self.num_layers, 2, -1, self.enc_hidden_size)
        # (num_layers, num_directions, batch_size, hidden_size)

        # Initialize a list to store the processed hidden states for each layer
        processed_hid = []

        for layer in range(self.num_layers):
            # Concatenate forward and backward hidden states for each layer
            hid_forward = hid[layer, 0, :, :]
            hid_backward = hid[layer, 1, :, :]
            concatenated_hid = torch.cat([hid_forward, hid_backward], dim=1)
            # concatenated_hid.shape = (batch_size, 2*hidden_size)

            # Apply the fully connected layer and activation function
            processed_hid.append(torch.tanh(self.fc(concatenated_hid)))

        # Stack the processed hidden states back to the shape (num_layers, batch_size, hidden_size)
        processed_hid = torch.stack(processed_hid, dim=0)
        # hid.shape = (num_layers, batch_size, dec_hidden_size)
        # out.shape = batch_size, seq_len, 2*enc_hidden_size)

        return out, processed_hid


# Dot-product Attention
# class Attention(nn.Module):
#     def __init__(self, enc_hidden_size, dec_hidden_size):
#         super(Attention, self).__init__()
#         self.enc_hidden_size = enc_hidden_size
#         self.dec_hidden_size = dec_hidden_size
#
#         self.linear_key = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
#         self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)
#
#     def forward(self, output, context, mask):
#         # mask.shape = (batch_size, output_len, context_len)
#         # output.shape =  (batch_size, output_len, dec_hidden_size)
#         # context.shape = (batch_size, context_len, 2*enc_hidden_size)
#
#         batch_size = output.size(0)
#         output_len = output.size(1)
#         input_len = context.size(1)  # input_len = context_len
#
#         # Key = W_k·context
#         context_key = self.linear_key(context.view(batch_size * input_len, -1)).view(
#             batch_size, input_len, -1)
#
#         # Attention score = Query · Key
#         # Query = output, Key = context_key
#         # context_key.transpose(1,2).shape = (batch_size, dec_hidden_size, context_len)
#         attn = torch.bmm(output, context_key.transpose(1, 2))
#         # attn.shape = (batch_size, output_len, context_len)
#
#         # Inhibit the effect of the specific location's attention score
#         mask = mask.to(attn.device)
#         attn.masked_fill(~mask, -1e6)
#
#         # Use softmax to calculate attention weight
#         attn = F.softmax(attn, dim=-1)
#         # attn.shape = (batch_size, output_len, context_len)
#
#         # Value = attention weight · context
#         context = torch.bmm(attn, context)
#         # context.shape = (batch_size, output_len, 2*enc_hidden_size)
#
#         # Concatenate the context vector and the hidden states to get the context representation
#         output = torch.cat((context, output), dim=-1)
#         # output.shape = (batch_size, output_len, 2*enc_hidden_size+dec_hidden_size)
#
#         # Calculate the output
#         output = output.view(batch_size * output_len, -1)
#         # output.shape = (batch_size*output_len, 2*enc_hidden_size+dec_hidden_size)
#         output = torch.tanh(self.linear_out(output))
#         # output.shape = (batch_size*output_len, dec_hidden_size)
#         output = output.view(batch_size, output_len, -1)
#         # output.shape = (batch_size, output_len, dec_hidden_size)
#         # attn.shape = (batch_size, output_len, context_len)
#         return output, attn


# Multiplicative Attention
class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_key = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_query = nn.Linear(dec_hidden_size, dec_hidden_size, bias=False)
        self.linear_value = nn.Linear(enc_hidden_size * 2, enc_hidden_size * 2, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # mask.shape = (batch_size, output_len, context_len)
        # output.shape =  (batch_size, output_len, dec_hidden_size)
        # context.shape = (batch_size, context_len, 2*enc_hidden_size)

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)  # input_len = context_len

        # Key = W_k·context
        context_key = self.linear_key(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)

        # Query = W_q·output
        output_query = self.linear_query(output.view(batch_size * output_len, -1)).view(
            batch_size, output_len, -1)

        # Value = W_value * context
        context_value = self.linear_value(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)

        # Attention score = Query · Key
        # Query = output_query, Key = context_key
        # output_query.shape = (batch_size, output_len, dec_hidden_size)
        # context_key.transpose(1,2).shape = (batch_size, dec_hidden_size, context_len)
        attn = torch.bmm(output_query, context_key.transpose(1, 2))
        # attn.shape = (batch_size, output_len, context_len)

        # Attention Normalization
        scaling_coef = 1.0 / torch.sqrt(torch.tensor(self.dec_hidden_size, dtype=torch.float32))
        attn = attn * scaling_coef

        # Inhibit the effect of the specific location's attention score
        mask = mask.to(attn.device)
        attn.masked_fill(~mask, -1e6)

        # Use softmax to calculate attention weight
        attn = F.softmax(attn, dim=-1)
        # attn.shape = (batch_size, output_len, context_len)

        # Value = attention weight · context
        context = torch.bmm(attn, context_value)
        # context.shape = (batch_size, output_len, 2*enc_hidden_size)

        # Concatenate the context vector and the hidden states to get the context representation
        output = torch.cat((context, output), dim=-1)
        # output.shape = (batch_size, output_len, 2*enc_hidden_size+dec_hidden_size)

        # Calculate the output
        output = output.view(batch_size * output_len, -1)
        # output.shape = (batch_size*output_len, 2*enc_hidden_size+dec_hidden_size)
        output = torch.tanh(self.linear_out(output))
        # output.shape = (batch_size*output_len, dec_hidden_size)
        output = output.view(batch_size, output_len, -1)
        # output.shape = (batch_size, output_len, dec_hidden_size)
        # attn.shape = (batch_size, output_len, context_len)
        return output, attn


# Additive  Attention
# class Attention(nn.Module):
#     def __init__(self, enc_hidden_size, dec_hidden_size):
#         super(Attention, self).__init__()
#         self.enc_hidden_size = enc_hidden_size
#         self.dec_hidden_size = dec_hidden_size
#
#         # Wa is the weight matrix for the linear transformation of concatenated [s_{i-1}; h_j]
#         self.Wa = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size, bias=False)
#         # va is the weight vector for the final dot product
#         self.va = nn.Linear(dec_hidden_size, 1, bias=False)
#         self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)
#
#     def forward(self, output, context, mask):
#         # output.shape = (batch_size, output_len, dec_hidden_size)
#         # context.shape = (batch_size, context_len, 2*enc_hidden_size)
#
#         batch_size = output.size(0)
#         output_len = output.size(1)
#         context_len = context.size(1)
#
#         # Repeat context_len times for each output_len
#         context = context.unsqueeze(1).repeat(1, output_len, 1, 1)
#         # context.shape = (batch_size, output_len, context_len, 2*enc_hidden_size)
#
#         output = output.unsqueeze(2).repeat(1, 1, context_len, 1)
#         # output.shape = (batch_size, output_len, context_len, dec_hidden_size)
#
#         # Concatenate output and context
#         combined = torch.cat((output, context), dim=-1)
#         # combined.shape = (batch_size, output_len, context_len, dec_hidden_size + 2*enc_hidden_size)
#
#         # Flatten combined for linear transformation
#         combined = combined.view(batch_size * output_len * context_len, -1)
#         # combined.shape = (batch_size * output_len * context_len, dec_hidden_size + 2*enc_hidden_size)
#
#         # Calculate energy scores
#         energy = torch.tanh(self.Wa(combined))
#         # energy.shape = (batch_size * output_len * context_len, dec_hidden_size)
#
#         # Calculate scores
#         score = self.va(energy).view(batch_size, output_len, context_len)
#         # score.shape = (batch_size, output_len, context_len)
#
#         # Apply mask
#         mask = mask.to(score.device)
#         score.masked_fill(~mask, -1e9)
#
#         # Calculate attention weights
#         attn_weights = F.softmax(score, dim=-1)
#         # attn_weights.shape = (batch_size, output_len, context_len)
#
#         # Calculate context vectors
#         context_vectors = torch.bmm(attn_weights, context.squeeze(1))
#         # context_vectors.shape = (batch_size, output_len, 2*enc_hidden_size)
#
#         # Concatenate the context vectors with the original output
#         output = torch.cat((context_vectors, output.squeeze(2)), dim=-1)
#         # output.shape = (batch_size, output_len, 2*enc_hidden_size + dec_hidden_size)
#
#         # Apply a linear layer and tanh activation to get the final output
#         output = torch.tanh(self.linear_out(output))
#         # output.shape = (batch_size, output_len, dec_hidden_size)
#
#         return output, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, dec_hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, output_len, input_len):
        # Generate a mask of shape output_len * input_len
        device = output_len.device
        max_output_len = output_len.max()
        max_input_len = input_len.max()

        output_mask = torch.arange(max_output_len, device=device)[None, :] < output_len[:, None]
        # output_mask.shape = (batch_size, output_len) # mask of output(English)
        input_mask = torch.arange(max_input_len, device=device)[None, :] < input_len[:, None]
        # input_mask.shape = (batch_size, context_len) # mask of input(Chinese)

        mask = (output_mask[:, :, None] & input_mask[:, None, :]).byte()
        mask = mask.bool()
        # output_mask[:, :, None].shape = (batch_size, output_len, 1)
        # input_mask[:, None, :].shape =  (batch_size, 1, context_len)
        # mask.shape = (batch_size, output_len, context_len)
        return mask

    def forward(self, encoder_context, x_lengths, y, y_lengths, hid):
        # x_lengths is the length of context vector(Chinese)
        # y_lengths is the length of standard translation vector(English)
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        # hid.shape = (num_layers, batch_size, dec_hidden_size)

        mask = self.create_mask(y_lengths, x_lengths)

        output, attn = self.attention(output_seq, encoder_context, mask)
        # output.shape = (batch_size, output_len, dec_hidden_size)
        # attn.shape = (batch_size, output_len, context_len)

        # Calculate the output probability of each word
        output = F.log_softmax(self.out(output), -1)
        # output.shape = (batch_size, output_len, vocab_size)
        return output, hid, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # Teacher Forcing
    def forward(self, enc_input, enc_input_lengths, dec_input, dec_input_lengths):
        encoder_context, hid = self.encoder(enc_input, enc_input_lengths)
        output, hid, attn = self.decoder(encoder_context=encoder_context,
                                         x_lengths=enc_input_lengths,
                                         y=dec_input,
                                         y_lengths=dec_input_lengths,
                                         hid=hid)
        # output.shape =(batch_size, output_len, vocab_size)
        # hid.shape = (num_layers, batch_size, dec_hidden_size)
        # attn.shape = (batch_size, output_len, context_len)
        return output, attn

    # Free Running
    def free_running(self, x, x_lengths, y, max_length):
        encoder_out, hid = self.encoder(x, x_lengths)
        output_by_step = []
        batch_size = x.shape[0]
        y = y.repeat(batch_size, 1)
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(encoder_out,
                                             x_lengths,
                                             y,
                                             torch.ones(batch_size).long().to(y.device),
                                             hid)
            y = output.max(2)[1].view(batch_size, 1)
            output_by_step.append(output)
            attns.append(attn)
        return torch.cat(output_by_step, 1), torch.cat(attns, 1)

    # Beam Search / Greedy Search(beam_width=1)
    def translate(self, x, x_lengths, start_token, end_token, beam_width=3, max_length=50):
        encoder_out, hid = self.encoder(x, x_lengths)

        # Initialize the beams
        start_tensor = torch.full((1, 1), start_token.item(), dtype=torch.long).to(x.device)
        beams = [(start_tensor, torch.tensor(0.0), hid, [])]
        completed_sequences = []

        for _ in range(max_length):
            new_beams = []
            for seq, score, hid, attn_weights in beams:
                y = seq[:, -1].view(1, 1)
                output, hid, attn = self.decoder(encoder_out,
                                                 x_lengths,
                                                 y,
                                                 torch.ones(1).long().to(y.device),
                                                 hid)

                # Get the top beam_width predictions
                topk_log_probs, topk_indices = torch.topk(F.log_softmax(output[:, -1, :], dim=-1), beam_width)

                for i in range(beam_width):
                    new_seq = torch.cat([seq, topk_indices[:, i].view(1, 1)], dim=-1)
                    new_score = score + topk_log_probs[:, i].item()
                    new_attn_weights = attn_weights + [attn.squeeze().cpu().detach().numpy()]
                    new_beams.append((new_seq, new_score, hid, new_attn_weights))

            # Sort the new beams and keep the top beam_width ones
            new_beams = sorted(new_beams, key=lambda x: x[1].item(), reverse=True)
            beams = new_beams[:beam_width]

            # Check if any of the beams has reached the end token
            indexes_to_remove = []
            for i, (seq, score, hid, attn_weights) in enumerate(beams):
                if seq[0, -1].item() == end_token.item():
                    completed_sequences.append((seq, score, attn_weights))
                    indexes_to_remove.append(i)

            for index in reversed(indexes_to_remove):
                del beams[index]

            if len(beams) == 0:
                break

        # If no sequence has reached the end token, use the best sequence in beams
        if not completed_sequences:
            completed_sequences = beams

        # Choose the sequence with the highest score
        best_seq, best_score, best_attn_weights = max(completed_sequences, key=lambda x: x[1].item())
        return best_seq[0, 1:], best_attn_weights
        # best_attn_weights.shape = (output_len, context_len)
