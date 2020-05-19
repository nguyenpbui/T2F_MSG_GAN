import torch as th
import os
import pickle
import numpy as np

def create_grid(samples, img_files, depth=7):
    
    from torchvision.utils import save_image
    from torch.nn.functional import interpolate
    from numpy import sqrt, power
    from MSG_GAN.GAN import Generator

    for i in range(len(samples)):
        samples[i] = interpolate(samples[i],
                                     scale_factor=power(2,
                                                        6 - i))
    # save the images:
    for sample, img_file in zip(samples, img_files):
        save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])),
                       normalize=True, scale_each=True, padding=0)

""" Example run"""
def example(trained_encoder, trained_ca, trained_gen, trained_dis):
	sample_dir = "training_runs/generated_samples/"
	with open('processed_annotations/processed_text.pkl', 'rb') as f:
		x = pickle.load(f)
	vocab_list = x['vocab']
	ListOfItems = vocab_list.items()

	device = th.device("cuda")
	input_text_string = input()
	input_text_raw = input_text_string.split(" ")
	input_text_tensor = [0] * 100
	for i in range(len(input_text_raw)):
		for item in ListOfItems:
			if item[1] == input_text_raw[i]:
				input_text_tensor[i]= item[0]
	input_text_tensor = np.array(input_text_tensor).reshape(1, 100)
	embeddings = trained_encoder(th.LongTensor(th.from_numpy(input_text_tensor)).to(device)).to(device)
	c_not_hats, mus, sigmas = trained_ca(embeddings)
	noise = th.randn(1, 512 - c_not_hats.shape[-1]).to(device)
	gan_input = th.cat((c_not_hats, noise), dim = -1).to(device)

	""" Generate sample image"""
	reses = [str(int(np.power(2, dep))) + "x"
	                         + str(int(np.power(2, dep)))
	                         for dep in range(2, 9)]

	gen_img_files = [os.path.join(sample_dir, res, "gen_" + ".png")
	                                 for res in reses]
	                    
	os.makedirs(sample_dir, exist_ok=True)

	for gen_img_file in gen_img_files:
	    os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)

	create_grid(samples=trained_gen(gan_input),
	                        img_files=gen_img_files)

def main():

	from networks.TextEncoder import Encoder
	from networks.ConditionAugmentation import ConditionAugmentor
	from MSG_GAN.GAN import Generator, Discriminator

	""" Load the trained model"""
	device = th.device("cuda")
	trained_encoder_path = "training_runs/saved_models/Encoder_98.pth"
	trained_ca_path = "training_runs/saved_models/Condition_Augmentor_98.pth"
	trained_gen_path = "training_runs/saved_models/GAN_GEN_98.pth"
	trained_dis_path = "training_runs/saved_models/GAN_DIS_98.pth"

	trained_encoder = Encoder(embedding_size=256, vocab_size=2763, hidden_size=512, num_layers=3, device="cuda")
	trained_encoder.load_state_dict(th.load(trained_encoder_path))
	trained_encoder.eval()

	trained_ca = ConditionAugmentor(input_size=512, latent_size=256, use_eql=True, device="cuda")
	trained_ca.load_state_dict(th.load(trained_ca_path))
	trained_ca.eval()

	trained_gen = Generator(depth=7, latent_size=512, dilation=1, use_spectral_norm=True).to(device)
	trained_gen.load_state_dict(th.load(trained_gen_path), strict=False)
	trained_gen.eval()
	
	trained_dis = Discriminator(depth=7, feature_size=512, dilation=1, use_spectral_norm=True)
	trained_dis.load_state_dict(th.load(trained_dis_path), strict=False)
	trained_dis.eval()

	example(trained_encoder, trained_ca, trained_gen, trained_dis)

if __name__ == '__main__':
	main()