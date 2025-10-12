import torch
import torch.nn as nn

# Original code from https://github.com/amanteur/BandSplitRNN-Pytorch
class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            group_num: int,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(group_num, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional # 输出是2*hidden_dim_size，因为是bi
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T      across K (keep in mind T->K, K->T)
        # print(x.shape)

        out = x.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]

        # print(out.shape)
        # print(self.groupnorm)
        out = self.groupnorm(
            out.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        out = self.rnn(out)[0]  # [BK, T, H]    [BT, K, H]， 最后一维是特征
        out = self.fc(out)  # [BK, T, N]    [BT, K, N]

        x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        return x


class BandSequenceModelModule(nn.Module):
    """
    BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n BiLSTMs in two dimensions - time and subbands.
    """

    def __init__(
            self,
            # group_num,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True,
            num_layers: int = 12,
            n_heads: int = 4,
    ):
        super(BandSequenceModelModule, self).__init__()

        self.bsrnn = nn.ModuleList([])
        self.n_heads = n_heads

        input_dim_size = input_dim_size // n_heads
        hidden_dim_size = hidden_dim_size // n_heads
        group_num = input_dim_size // 16
        # print(f"input_dim_size: {input_dim_size}, hidden_dim_size: {hidden_dim_size}, group_num: {group_num}")

        # print(group_num, input_dim_size)

        for _ in range(num_layers):
            rnn_across_t = RNNModule(
                group_num, input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            rnn_across_k = RNNModule(
                group_num, input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            self.bsrnn.append(
                nn.Sequential(rnn_across_t, rnn_across_k)
            )

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        # x (b,c,t,f)
        b,c,t,f = x.shape
        x = x.view(b * self.n_heads, c // self.n_heads, t, f) # [b*n_heads, c//n_heads, t, f]

        x = x.permute(0, 3, 2, 1).contiguous()  # [b*n_heads, f, t, c//n_heads]
        for i in range(len(self.bsrnn)):
            x = self.bsrnn[i](x)

        x = x.permute(0, 3, 2, 1).contiguous()  # [b*n_heads, c//n_heads, t, f]
        x = x.view(b, c, t, f)  # [b, c, t, f]
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, k_subbands, t_timesteps, input_dim = 4, 41, 512, 128
    in_features = torch.rand(batch_size, k_subbands, t_timesteps, input_dim).to(device)

    cfg = {
        # "t_timesteps": t_timesteps,
        "group_num": 32,
        "input_dim_size": 128,
        "hidden_dim_size": 256,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 1
    }
    model = BandSequenceModelModule(**cfg).to(device)
    _ = model.eval()

    with torch.no_grad():
        out_features = model(in_features)

    print(f"In: {in_features.shape}\nOut: {out_features.shape}")
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")
