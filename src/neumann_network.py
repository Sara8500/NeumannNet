from src.neuralnet import RegularizerNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class NeumannNetwork:

    def __init__(self, learning_rate, forward_adjoint, forward_gramian, corruption_model, num_blocks):

        self.learning_rate = learning_rate
        self.forward_adjoint = forward_adjoint
        self.forward_gramian = forward_gramian
        self.corruption_model = corruption_model
        self.eta = 0.1 #fixme, make trainable?
        self.B = num_blocks

        self.netR = RegularizerNet()

        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'

        self.netR.to(device)

        if device == 'cuda':
            print("cuda available.")
            cudnn.benchmark = True

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.netR.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

    def pass_through_net(self, y_tensor):
        # initialize runner
        runner = self.eta * self.forward_adjoint(y_tensor)
        neumann_sum = runner


        # run through B blocks
        for i in range(0, self.B):
            linear_component = runner - self.eta * self.forward_gramian(runner)
            regularizer_output = self.netR.forward(runner)
            learned_component = -regularizer_output

            runner = linear_component + learned_component
            neumann_sum = neumann_sum + runner

        return neumann_sum

    def train(self, dataloader):

        for batch_id, (data, _) in enumerate(dataloader):

            data = data.to(device)

            corrupted_blurred_data = self.corruption_model(self.forward_adjoint(data))

            self.optimizer.zero_grad()

            output = self.pass_through_net(corrupted_blurred_data)

            loss = self.criterion(output, data)
            print("loss: ", loss)
            loss.backward()
            self.optimizer.step()

    def test(self, dataloader, n_batches):

        storage_test_images = np.zeros(shape=(dataloader.batch_size * n_batches, 3, 32, 32))
        storage_distorted_images = np.zeros(shape=(dataloader.batch_size * n_batches, 3, 32, 32))
        storage_reconstructed_images = np.zeros(shape=(dataloader.batch_size * n_batches, 3, 32, 32))

        for batch_id, (data, _) in enumerate(dataloader):       # Batchsize should be 1 here

            print("batch id: ", batch_id)

            if batch_id >= n_batches:
                break

            storage_test_images[batch_id, :, :, :] = data.numpy()[0, :, :, :]

            data = data.to(device)

            corrupted_data = self.corruption_model(self.forward_adjoint(data))
            storage_distorted_images[batch_id, :, :, :] = corrupted_data.numpy()[0, :, :, :]
            output = self.pass_through_net(corrupted_data)
            storage_reconstructed_images[batch_id, :, :, :] = output.detach().numpy()[0, :, :, :]

            loss = self.criterion(output, data)
            print("testing loss: ", loss)

        return storage_test_images, storage_distorted_images, storage_reconstructed_images, loss

