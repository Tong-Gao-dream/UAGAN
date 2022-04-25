import torch
from torch.autograd import Variable


def train(args, dataloader, generator, discriminator, optim_G, optim_D, loss_adv, loss_rec):
    for epoch in range(args.epoch):
        for i_batch, sample_batched in enumerate(dataloader):
            # update generator

            img, mask = \
                sample_batched['image'], sample_batched['mask']

            valid = Variable(torch.cuda.FloatTensor(mask.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False)

            valid = valid.cuda()
            fake = fake.cuda()

            mask = mask.cuda()
            img = img.cuda()

            optim_G.zero_grad()

            g_output = generator(img)

            loss_adv_ = loss_adv(discriminator(g_output), valid)

            mask = mask.float()

            loss_rec_ = loss_rec(g_output, mask)
            g_loss = (loss_adv_ + loss_rec_) / 2

            g_loss.backward()
            optim_G.step()

            # update discriminator

            optim_D.zero_grad()

            # print('discriminator(mask)', discriminator(mask).shape)
            # print('valid', valid.shape)
            real_loss = loss_adv(discriminator(mask), valid)
            fake_loss = loss_adv(discriminator(g_output.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward(retain_graph=True)
            optim_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epoch, i_batch, len(dataloader), d_loss.item(), g_loss.item())
            )
        generator.eval()
        torch.save(generator.state_dict(), './save_model/save_G/UAGAN_generator.pth')
        torch.save(discriminator.state_dict(), './save_model/save_D/UAGAN_discriminator.pth')
