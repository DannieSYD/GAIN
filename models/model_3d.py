import torch
import time


class Model3D(torch.nn.Module):
    def __init__(self, model_factory, hidden_dim, out_dim, device, num_parts=1):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_factory() for _ in range(num_parts)])
        self.linear = torch.nn.Linear(hidden_dim * num_parts, out_dim)
        self.device = device

    def forward(self, batched_data):
        outs = []
        # print('outs = []')

        for model, data in zip(self.models, batched_data):
            # print(f"{time.time()}:for model, data in zip(self.models, batched_data):")
            data = data.to(self.device)
            # print(f"{time.time()}:data = data.to(self.device)")
            z, pos, batch = data.x[:, 0], data.pos, data.batch
            # print(f"{time.time()}:z, pos, batch = data.x[:, 0], data.pos, data.batch")
            out = model(z, pos, batch)
            # print(f"{time.time()}:out = model(z, pos, batch)")
            if model.__class__.__name__ == 'LEFTNet':
                out = out[0]
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        outs = self.linear(outs).squeeze(-1)
        return outs

