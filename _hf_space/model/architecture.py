"""
GuitarTabCRNN — SoloTab モデルアーキテクチャ（HF Space版）
"""
import torch
import torch.nn as nn
import config


class TabCNN(nn.Module):
    def __init__(
        self,
        input_channels=config.CNN_INPUT_CHANNELS,
        output_channels_list=None,
        kernel_sizes=None,
        strides=None,
        paddings=None,
        pooling_kernels=None,
        pooling_strides=None,
    ):
        super().__init__()

        oc_list = output_channels_list or config.CNN_OUTPUT_CHANNELS_LIST_DEFAULT
        ks_list = kernel_sizes or config.CNN_KERNEL_SIZES_DEFAULT
        s_list = strides or config.CNN_STRIDES_DEFAULT
        p_list = paddings or config.CNN_PADDINGS_DEFAULT
        pk_list = pooling_kernels or config.CNN_POOLING_KERNELS_DEFAULT
        ps_list = pooling_strides or config.CNN_POOLING_STRIDES_DEFAULT

        num_layers = len(oc_list)
        self.conv_layers = nn.ModuleList()
        current_channels = input_channels
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, oc_list[i],
                              kernel_size=ks_list[i], stride=s_list[i], padding=p_list[i]),
                    nn.BatchNorm2d(oc_list[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pk_list[i], stride=ps_list[i]),
                )
            )
            current_channels = oc_list[i]
        self.output_channels = current_channels

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class GuitarTabCRNN(nn.Module):
    def __init__(
        self,
        num_frames_rnn_input_dim,
        rnn_type="LSTM",
        rnn_hidden_size=256,
        rnn_layers=2,
        rnn_dropout=0.2,
        rnn_bidirectional=False,
        num_strings=config.DEFAULT_NUM_STRINGS,
        max_frets_val=config.MAX_FRETS,
    ):
        super().__init__()
        self.num_strings = num_strings
        self.num_fret_classes = max_frets_val + config.FRET_SILENCE_CLASS_OFFSET + 1

        self.cnn = TabCNN()
        self.rnn_input_dim = num_frames_rnn_input_dim

        rnn_params = {
            "input_size": self.rnn_input_dim,
            "hidden_size": rnn_hidden_size,
            "num_layers": rnn_layers,
            "batch_first": True,
            "bidirectional": rnn_bidirectional,
            "dropout": rnn_dropout if rnn_layers > 1 else 0,
        }

        if rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(**rnn_params)
        elif rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(**rnn_params)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        rnn_output_size = (2 * rnn_hidden_size) if rnn_bidirectional else rnn_hidden_size
        self.onset_fc = nn.Linear(rnn_output_size, self.num_strings)
        self.fret_fc = nn.Linear(rnn_output_size, self.num_strings * self.num_fret_classes)

    def forward(self, x):
        x_cnn = self.cnn(x)
        batch_size, channels_out, reduced_n_mels, reduced_n_frames = x_cnn.shape
        x_rnn_input = x_cnn.permute(0, 3, 1, 2)
        x_rnn_input = x_rnn_input.reshape(batch_size, reduced_n_frames, channels_out * reduced_n_mels)
        x_rnn_output, _ = self.rnn(x_rnn_input)
        onset_logits = self.onset_fc(x_rnn_output)
        fret_logits_flat = self.fret_fc(x_rnn_output)
        fret_logits = fret_logits_flat.reshape(
            batch_size, reduced_n_frames, self.num_strings, self.num_fret_classes
        )
        return onset_logits, fret_logits
