""" Script for end-to-end training of the T2F model """
import datetime
import time
import torch as th
import numpy as np
import data_processing.DataLoader as dl
import argparse
import yaml
import os
import pickle
import timeit

from torch.backends import cudnn
from torch.nn.functional import avg_pool2d
# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# set torch manual seed for consistent output
th.manual_seed(3)

# Start fast training mode:
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", type=str, default="configs/2.conf",
                        help="default configuration for the Network")
    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")
    parser.add_argument("--encoder_file", action="store", type=str, default=None,
                        help="pretrained Encoder file (compatible with my code)")
    parser.add_argument("--ca_file", action="store", type=str, default=None,
                        help="pretrained Conditioning Augmentor file (compatible with my code)")
    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")

    args = parser.parse_args()

    return args


def get_config(conf_file):

    from easydict import EasyDict as edict

    with open(conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)

def create_grid_(samples, scale_factor, img_file, real_imgs=False):
    """
    utility function to create a grid of GAN samples
    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :param real_imgs: turn off the scaling of images
    :return: None (saves a file)
    """
    from torchvision.utils import save_image
    from torch.nn.functional import interpolate

    samples = th.clamp((samples / 2) + 0.5, min=0, max=1)

    # upsample the image
    if not real_imgs and scale_factor > 1:
        samples = interpolate(samples,
                              scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=int(np.sqrt(len(samples))))
    
def create_grid(samples, img_files, depth=7):
    """
    utility function to create a grid of GAN samples
    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :param real_imgs: turn off the scaling of images
    :return: None (saves a file)
    """
    from torchvision.utils import save_image
    from torch.nn.functional import interpolate
    from numpy import sqrt, power
    from MSG_GAN.GAN import Generator

        # dynamically adjust the colour of the images
    #samples = [Generator.adjust_dynamic_range(sample) for sample in samples]

        # resize the samples to have same resolution:
    for i in range(len(samples)):
        samples[i] = interpolate(samples[i],
                                     scale_factor=power(2,
                                                        6 - i))
        # save the images:
    for sample, img_file in zip(samples, img_files):
        save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])),
                       normalize=True, scale_each=True, padding=0)

def create_descriptions_file(file, captions, dataset):

    from functools import reduce

    # transform the captions to text:
    if isinstance(captions, th.Tensor):
        captions = list(map(lambda x: dataset.get_english_caption(x.cpu()),
                            [captions[i] for i in range(captions.shape[0])]))

        with open(file, "w") as filler:
            for caption in captions:
                filler.write(reduce(lambda x, y: x + " " + y, caption, ""))
                filler.write("\n\n")
    else:
        with open(file, "w") as filler:
            for caption in captions:
                filler.write(caption)
                filler.write("\n\n")

def train_networks(encoder, ca, msg_gan, dataset,
                   encoder_optim, ca_optim, gen_optim, dis_optim, loss_fn, fade_in_percentage,
                   batch_size, start_depth, num_workers, feedback_factor,
                   log_dir, sample_dir, checkpoint_factor, num_epochs,
                   save_dir, use_matching_aware_dis=True):
    # required only for type checking
    from networks.TextEncoder import PretrainedEncoder
    from numpy import power

    # put all the Networks in training mode:
    ca.train()
    msg_gan.gen.train()
    msg_gan.dis.train()

    if not isinstance(encoder, PretrainedEncoder):
        encoder.train()

    print("Starting the training process ... ")

    # create fixed_input for debugging###################################################
    temp_data = dl.get_data_loader(dataset, batch_size, num_workers=8)
    fixed_captions, fixed_real_images = iter(temp_data).next()
    fixed_embeddings = encoder(fixed_captions.to(device)).to(device)

    fixed_c_not_hats, _, _ = ca(fixed_embeddings)

    fixed_noise = th.randn(len(fixed_captions),
                           msg_gan.latent_size - fixed_c_not_hats.shape[-1]).to(device)

    fixed_gan_input = th.cat((fixed_c_not_hats, fixed_noise), dim=-1)
    fixed_save_dir = os.path.join(sample_dir, "__Real_Info")
    os.makedirs(fixed_save_dir, exist_ok=True)
    create_grid_(fixed_real_images, None,  # scale factor is not required here
                os.path.join(fixed_save_dir, "real_samples.png"), real_imgs=True)
    create_descriptions_file(os.path.join(fixed_save_dir, "real_captions.txt"),
                             fixed_captions,
                             dataset)
    # create a global time counter
    global_time = time.time()

    # delete temp data loader:
    del temp_data
    ####################################################################################
    ####################################################################################
    data = dl.get_data_loader(dataset, batch_size, num_workers)

    ticker = 1

    for epoch in range(1, num_epochs + 1):
        start = timeit.default_timer()  # record time at the start of epoch
        print("\nEpoch: %d" % epoch)
        total_batches = len(iter(data))
        limit = int((fade_in_percentage / 100) * total_batches)
  
        for (i, batch) in enumerate(data, 1):
            # extract current batch of data for training
            captions, images = batch
            images = images.to(device)
            extracted_batch_size = images.shape[0]
            if encoder_optim is not None:
                captions = captions.to(device)

            #create a list of downsampled images from the real images:
            images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                                     for i in range(1, 7)]
            images = list(reversed(images))
            # perform text_work:
            embeddings = encoder(captions).to(device)
            #embeddings = embeddings.detach()
            c_not_hats, mus, sigmas = ca(embeddings)
            #add noise to dis loss
            z = th.randn(
                extracted_batch_size,
                msg_gan.latent_size - c_not_hats.shape[-1]
            ).to(device)

            gan_input = th.cat((c_not_hats, z), dim=-1)
            # optimize the discriminator:
            dis_loss = msg_gan.optimize_discriminator(dis_optim, gan_input, images,
                                                            loss_fn)
            #add noise to gen loss
            z = th.randn(
                captions.shape[0] if isinstance(captions, th.Tensor) else len(captions),
                msg_gan.latent_size - c_not_hats.shape[-1]
            ).to(device)

            gan_input = th.cat((c_not_hats, z), dim=-1)

            if encoder_optim is not None:
                encoder_optim.zero_grad()

            ca_optim.zero_grad()

            gen_loss = msg_gan.optimize_generator(gen_optim, gan_input, images,
                                                           loss_fn)
            gen_loss = msg_gan.optimize_generator(gen_optim, gan_input, images,
                                                        loss_fn)
            gen_loss = msg_gan.optimize_generator(gen_optim, gan_input, images,
                                                        loss_fn)
            # once the optimize_generator is called, it also sends gradients
            # to the Conditioning Augmenter and the TextEncoder. Hence the
            # zero_grad statements prior to the optimize_generator call
            # now perform optimization on those two as well
            # obtain the loss (KL divergence from ca_optim)
            kl_loss = th.mean(0.5 * th.sum((mus ** 2) + (sigmas ** 2)
                                               - th.log((sigmas ** 2)) - 1, dim=1))
            kl_loss.backward(retain_graph=True)
            ca_optim.step()
            if encoder_optim is not None:
                encoder_optim.step()

                # provide a loss feedback
            if i % int(limit / feedback_factor) == 0 or i == 1:
                elapsed = time.time() - global_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [%s]  batch: %d  d_loss: %f  g_loss: %f  kl_los: %f"
                      % (elapsed, i, dis_loss, gen_loss, kl_loss.item()))

                    # also write the losses to the log file:
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, "loss.log")
                with open(log_file, "a") as log:
                    log.write(str(dis_loss) + "\t" + str(gen_loss)
                              + "\t" + str(kl_loss.item()) + "\n")

                    # create a grid of samples and save it
                reses = [str(int(np.power(2, dep))) + "x"
                         + str(int(np.power(2, dep)))
                         for dep in range(2, 9)]

                gen_img_files = [os.path.join(sample_dir, res, "gen_" +
                                                  str(epoch) + "_" +
                                              str(i) + ".png")
                                 for res in reses]
                    
                os.makedirs(sample_dir, exist_ok=True)

                for gen_img_file in gen_img_files:
                    os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)

                dis_optim.zero_grad()
                gen_optim.zero_grad()

                with th.no_grad():

                    create_grid(samples=msg_gan.gen(fixed_gan_input),
                        img_files=gen_img_files
                        )
                th.cuda.empty_cache()
            # increment the ticker:
            ticker += 1

        stop = timeit.default_timer()
        print("Time taken for epoch: %.3f secs" % (stop - start))

        if epoch % checkpoint_factor == 0 or epoch == 0:
            # save the Model
            encoder_save_file = os.path.join(save_dir, "Encoder_" +
                                             str(epoch) + ".pth")
            ca_save_file = os.path.join(save_dir, "Condition_Augmentor_" +
                                        str(epoch) + ".pth")
            gen_save_file = os.path.join(save_dir, "GAN_GEN_" +
                                         str(epoch) + ".pth")
            dis_save_file = os.path.join(save_dir, "GAN_DIS_" +
                                         str(epoch) + ".pth")

            os.makedirs(save_dir, exist_ok=True)

            if encoder_optim is not None:
                th.save(encoder.state_dict(), encoder_save_file, pickle)
            th.save(ca.state_dict(), ca_save_file, pickle)
            th.save(msg_gan.gen.module.state_dict(), gen_save_file, pickle)
            th.save(msg_gan.dis.module.state_dict(), dis_save_file, pickle)

    print("Training completed ...")
    ca.eval()
    encoder.eval()
    msg_gan.gen.eval()
    msg_gan.dis.eval()

def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from networks.TextEncoder import Encoder
    from networks.ConditionAugmentation import ConditionAugmentor
    from MSG_GAN.GAN import MSG_GAN
    from MSG_GAN import Losses as lses

    print(args.config)
    config = get_config(args.config)
    print("Current Configuration:", config)

    # create the dataset for training
    if config.use_pretrained_encoder:
        dataset = dl.RawTextFace2TextDataset(
            annots_file=config.annotations_file,
            img_dir=config.images_dir,
            img_transform=dl.get_transform(config.img_dims)
        )
        from networks.TextEncoder import PretrainedEncoder
        # create a new session object for the pretrained encoder:
        text_encoder = PretrainedEncoder(
            model_file=config.pretrained_encoder_file,
            embedding_file=config.pretrained_embedding_file,
            device=device
        )
        encoder_optim = None
    else:
        dataset = dl.Face2TextDataset(
            pro_pick_file=config.processed_text_file,
            img_dir=config.images_dir,
            img_transform=dl.get_transform(config.img_dims),
            captions_len=config.captions_length
        )
        text_encoder = Encoder(
            embedding_size=config.embedding_size,
            vocab_size=dataset.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            device=device
        )
        encoder_optim = th.optim.Adam(text_encoder.parameters(),
                                      lr=config.learning_rate,
                                      betas=(config.adam_beta1, config.adam_beta2),
                                      eps=config.eps)
    msg_gan = MSG_GAN(
        device=device)

    genoptim = th.optim.Adam(msg_gan.gen.parameters(), config.g_lr,
                              [config.adam_beta1, config.adam_beta2])

    disoptim = th.optim.Adam(msg_gan.dis.parameters(), config.d_lr,
                              [config.adam_beta1, config.adam_beta2])

    loss = lses.RelativisticAverageHingeGAN

    # create the networks

    if args.encoder_file is not None:
        # Note this should not be used with the pretrained encoder file
        print("Loading encoder from:", args.encoder_file)
        text_encoder.load_state_dict(th.load(args.encoder_file))

    condition_augmenter = ConditionAugmentor(
        input_size=config.hidden_size,
        latent_size=config.ca_out_size,
        use_eql=config.use_eql,
        device=device
    )

    if args.ca_file is not None:
        print("Loading conditioning augmenter from:", args.ca_file)
        condition_augmenter.load_state_dict(th.load(args.ca_file))

    if args.generator_file is not None:
        print("Loading generator from:", args.generator_file)
        msg_gan.gen.load_state_dict(th.load(args.generator_file))

    if args.discriminator_file is not None:
        print("Loading discriminator from:", args.discriminator_file)
        msg_gan.dis.load_state_dict(th.load(args.discriminator_file))

    # create the optimizer for Condition Augmenter separately
    ca_optim = th.optim.Adam(condition_augmenter.parameters(),
                             lr=config.learning_rate,
                             betas=(config.adam_beta1, config.adam_beta2),
                             eps=config.eps)

    print("Generator Config:")
    print(msg_gan.gen)

    print("\nDiscriminator Config:")
    print(msg_gan.dis)

    # train all the networks
    train_networks(
        encoder=text_encoder,
        ca=condition_augmenter,
        msg_gan=msg_gan,
        dataset=dataset,
        encoder_optim=encoder_optim,
        ca_optim=ca_optim,
        gen_optim=genoptim,
        dis_optim=disoptim,
        loss_fn=loss(device,msg_gan.dis),
        fade_in_percentage=config.fade_in_percentage,
        start_depth=args.start_depth,
        batch_size=config.batch_size,
        num_workers=8,
        feedback_factor=config.feedback_factor,
        log_dir=config.log_dir,
        sample_dir=config.sample_dir,
        checkpoint_factor=config.checkpoint_factor,
        save_dir=config.save_dir,
        num_epochs=config.num_epochs
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
