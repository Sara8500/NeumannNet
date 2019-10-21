import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from src.MoDL_utils import cg_pseudoinverse


class PreconditionedNeumannNet:
    def __init__(self, learning_rate, forward_adjoint, forward_gramian, corruption_model, num_blocks):

        self.learning_rate = learning_rate
        self.forward_adjoint = forward_adjoint
        self.forward_gramian = forward_gramian
        self.corruption_model = corruption_model
        self.B = num_blocks
        self.netR = RegularizerNet()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.netR.to(device)

        if device == 'cuda':
            self.netR = torch.nn.DataParallel(self.netR)
            cudnn.benchmark = True

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.netR.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        def pass_through_net(self, y_tensor):

            # print("y", y_tensor.shape)

            # initialize runner
            runner = self.eta * self.forward_adjoint(y_tensor)
            neumann_sum = runner

            # run through B blocks
            for i in range(0, self.B):
                linear_component = runner - self.eta * blur_gramian(runner)
                regularizer_output = self.netR(runner)
                learned_component = -regularizer_output

                runner = linear_component + learned_component
                neumann_sum = neumann_sum + runner

            return neumann_sum

        def train(self, data_loader):

            for batch_id, (data, _) in enumerate(data_loader):
                data = data.to(device)

                corrupted_blurred_data = self.corruption_model(self.forward_adjoint(data))

                self.optimizer.zero_grad()
                output = self.pass_through_net(corrupted_blurred_data)
                loss = self.criterion(output, data)
                print("loss: ", loss)
                loss.backward()
                self.optimizer.step()

