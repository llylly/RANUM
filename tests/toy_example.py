import torch


if __name__ == '__main__':
    x_input = torch.tensor([5.635,-5.635], requires_grad=True)
    y_input = torch.tensor([1.,0.])
    W = torch.tensor([[-0.1,0.1],[0.1,-0.1]], requires_grad=True)
    b = torch.tensor([-0.0, 0.0], requires_grad=True)
    model_output = x_input @ W + b
    model_output = torch.softmax(model_output, dim=0)
    ans = - torch.min(y_input * torch.log(model_output) + (1.0 - y_input) * torch.log(1.0 - model_output))
    ans.backward()
    print(-W.grad)
    print(-b.grad)

    print('new_W:', W-W.grad)
    print('new_b:', b-b.grad)

    print(torch.softmax(torch.tensor([40., -40.]), dim=0))

    print(torch.log(torch.softmax(torch.tensor([44.8, -44.8], dtype=torch.float32), dim=0)))
    print(torch.log(torch.softmax(torch.tensor([55, -55], dtype=torch.float32), dim=0)))