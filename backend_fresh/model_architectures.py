import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertConfig
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MotionBERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = nn.Linear(3, config.hidden_size)  # 3D coordinates to hidden dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation=config.hidden_act,
                batch_first=True
            ),
            num_layers=config.num_hidden_layers
        )
        
        self.init_weights()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 3)
        x = self.embeddings(x)  # Convert to hidden dim
        x = self.encoder(x)
        return x

class Pose3DModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_joints=17, output_dims=3):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_joints * output_dims, 3)
        self.output_joints = output_joints
        self.output_dims = output_dims

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x)
        return x.view(batch_size, self.output_joints, self.output_dims)

class ActionRecognitionModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=60):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, num_classes, 3)

    def forward(self, x):
        return self.mlp(x)

class MeshModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_vertices=6890, vertex_dims=3):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, num_vertices * vertex_dims, 3)
        self.num_vertices = num_vertices
        self.vertex_dims = vertex_dims

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x)
        return x.view(batch_size, self.num_vertices, self.vertex_dims)
