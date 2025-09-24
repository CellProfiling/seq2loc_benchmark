import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_hidden_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),  # Added this bc it is what LAProtT5 did
        )
        self.hidden = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),  # Added this bc it is what LAProtT5 did
                )
            )

        self.classifer = nn.Linear(hidden_dim, num_classes)  # output logits

    def forward(self, embeddings):
        output = self.input(embeddings)
        for hidden_layer in self.hidden:
            output = hidden_layer(output)
        output = self.classifer(output)
        return output
