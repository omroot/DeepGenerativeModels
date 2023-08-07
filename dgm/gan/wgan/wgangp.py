
import torch
from torch import nn
from tqdm.auto import tqdm


from dgm.gan.wgan.wgangp_critic import Critic
from dgm.gan.dcgan.dcgan_generator import DCGAN_Generator
class WGANGP():

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
        self.discriminator = Critic().to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(beta_1, beta_2))
        self.test_generator = test_generator
        self.real_data = None
        self.display_step =display_step
        pass

    def fit(self, dataloader):

        self.generator = self.generator.apply(self.weights_init)
        self.discriminator = self.discriminator.apply(self.weights_init)

        cur_step = 0
        generator_losses = []
        critic_losses = []
        for epoch in range(self.n_epochs):
            # Dataloader returns the batches
            for real, _ in tqdm(dataloader):
                cur_batch_size = len(real)
                real = real.to(self.device)

                mean_iteration_critic_loss = 0
                for _ in range(crit_repeats):
                    ### Update critic ###
                    crit_opt.zero_grad()
                    fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                    fake = gen(fake_noise)
                    crit_fake_pred = crit(fake.detach())
                    crit_real_pred = crit(real)

                    epsilon = torch.rand(len(real), 1, 1, 1, device=self.device, requires_grad=True)
                    gradient = get_gradient(crit, real, fake.detach(), epsilon)
                    gp = gradient_penalty(gradient)
                    crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                    # Keep track of the average critic loss in this batch
                    mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                    # Update gradients
                    crit_loss.backward(retain_graph=True)
                    # Update optimizer
                    crit_opt.step()
                critic_losses += [mean_iteration_critic_loss]

                ### Update generator ###
                gen_opt.zero_grad()
                fake_noise_2 = get_noise(cur_batch_size, z_dim, device=self.device)
                fake_2 = gen(fake_noise_2)
                crit_fake_pred = crit(fake_2)

                gen_loss = get_gen_loss(crit_fake_pred)
                gen_loss.backward()

                # Update the weights
                gen_opt.step()

                # Keep track of the average generator loss
                generator_losses += [gen_loss.item()]

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    gen_mean = sum(generator_losses[-display_step:]) / display_step
                    crit_mean = sum(critic_losses[-display_step:]) / display_step
                    print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                    show_tensor_images(fake)
                    show_tensor_images(real)
                    step_bins = 20
                    num_examples = (len(generator_losses) // step_bins) * step_bins
                    plt.plot(
                        range(num_examples // step_bins),
                        torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                        label="Generator Loss"
                    )
                    plt.plot(
                        range(num_examples // step_bins),
                        torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                        label="Critic Loss"
                    )
                    plt.legend()
                    plt.show()

                cur_step += 1


