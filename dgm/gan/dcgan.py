

import torch
from torch import nn
from tqdm.auto import tqdm


from dgm.gan.dcgan_generator import DCGAN_Generator, get_noise
from dgm.gan.dcgan_discriminator import DCGAN_Discriminator

from dgm.utils.visualization import show_tensor_images

class DCGAN():

    def __init__(self,
                 noise_dimension,
                 loss_function,
                 number_epochs,
                 batch_size,
                 learning_rate,
                 beta_1,
                 beta_2,
                 display_step,
                 test_generator = True,
                 device = 'mps'
                 ):
        """
        Constructor of the GAN class
        Parameters
        :param noise_dimension: dimension of the
        :param loss_function:
        :param number_epochs:
        :param batch_size:
        :param learning_rate:
        :param device:
        """
        if loss_function == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        self.n_epochs = number_epochs
        self.noise_dimension = noise_dimension
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device = device
        self.generator = DCGAN_Generator(noise_dimension).to(device)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(beta_1, beta_2))
        self.discriminator = DCGAN_Discriminator().to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(beta_1, beta_2))
        self.test_generator = test_generator
        self.real_data = None
        self.display_step =display_step
        pass

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


    def get_discriminator_loss(self, real_data):

        ''' Return the loss of the discriminator given inputs.
        These are the steps needed to compute the loss:
              1) Create noise vectors and generate a batch (num_images) of fake images.
                   Make sure to pass the device argument to the noise.
              2) Get the discriminator's prediction of the fake image
                   and calculate the loss. Don't forget to detach the generator!
              3) Get the discriminator's prediction of the real image and calculate the loss.
              4) Calculate the discriminator's loss by averaging the real and fake loss
                   and set it to disc_loss.

        Parameters:
            real: a batch of real data points
        Returns:
            discriminator_loss: a torch scalar loss value for the current batch
        '''
        sample_size = len(real_data)

        fake_noise = get_noise(sample_size, self.noise_dimension, device=self.device)
        fake_data = self.generator(fake_noise)
        discriminator_fake_prediction = self.discriminator(fake_data.detach())
        discriminator_fake_loss = self.criterion(discriminator_fake_prediction, torch.zeros_like(discriminator_fake_prediction))
        discriminator_real_prediction = self.discriminator(real_data)
        discriminator_real_loss = self.criterion(discriminator_real_prediction, torch.ones_like(discriminator_real_prediction))
        discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) / 2

        return discriminator_loss

    def get_generator_loss(self, sample_size):
        ''' Return the loss of the generator given inputs.
        The steps needed to complete the computation are:
                1) Create noise vectors and generate a batch of fake images.
                   Remember to pass the device argument to the get_noise function.
                2) Get the discriminator's prediction of the fake image.
                3) Calculate the generator's loss. Remember the generator wants
        Parameters:
            sample_size: the size of the fake data to be generated
        Returns:
            generator_loss: a torch scalar loss value for the current batch
        '''
        fake_noise = get_noise(sample_size, self.noise_dimension, device=self.device)
        fake_data = self.generator(fake_noise)
        discriminator_fake_pred = self.discriminator(fake_data)
        generator_loss = self.criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))
        return generator_loss

    def fit(self, dataloader):

        self.generator = self.generator.apply(self.weights_init)
        self.discriminator = self.discriminator.apply(self.weights_init)

        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        gen_loss = False
        error = False

        for epoch in range(self.n_epochs):

            # Dataloader returns the batches
            for real, _ in tqdm(dataloader):
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.view(cur_batch_size, -1).to(self.device)

                ### Update discriminator ###
                # Zero out the gradients before backpropagation
                self.discriminator_optimizer.zero_grad()

                # Calculate discriminator loss
                disc_loss = self.get_discriminator_loss(real)

                # Update gradients
                disc_loss.backward(retain_graph=True)

                # Update optimizer
                self.discriminator_optimizer.step()

                # For testing purposes, to keep track of the generator weights
                if self.test_generator:
                    old_generator_weights = self.generator.gen[0][0].weight.detach().clone()
                ### Update generator ###
                # Zero out the gradients.
                self.generator_optimizer.zero_grad()
                #   Calculate the generator loss, assigning it to gen_loss.
                gen_loss = self.get_generator_loss(cur_batch_size)
                #  update the gradients
                gen_loss.backward(retain_graph=True)
                # Update  optimizer.
                self.generator_optimizer.step()

                # For testing purposes, to check that your code changes the generator weights
                if self.test_generator:
                    try:
                        assert self.lr > 0.0000002 or (self.generator.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                        assert torch.any(self.generator.gen[0][0].weight.detach().clone() != old_generator_weights)
                    except:
                        error = True
                        print("Runtime tests have failed")

                ### Visualization code ###
                if cur_step % self.display_step == 0 and cur_step > 0:
                    print(
                        f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    fake_noise = get_noise(cur_batch_size, self.noise_dimension, device=self.device)
                    fake = self.generator(fake_noise)
                    show_tensor_images(fake)
                    show_tensor_images(real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                cur_step += 1
        self.real_data = real

        pass


    def generate(self, sample_size):
        fake_noise = get_noise(sample_size, self.noise_dimension, device=self.device)
        fake_data = self.generator(fake_noise)
        return fake_data



