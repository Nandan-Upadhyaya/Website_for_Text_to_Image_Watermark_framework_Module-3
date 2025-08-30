import os
import sys
import time
import random
import numpy as np
import torch
from pathlib import Path
import nltk

# Ensure DF-GAN code is in the path
DF_GAN_PATH = Path(os.getenv('DF_GAN_PATH', r'C:\Users\Tejas\Desktop\stl\github\DF-GAN'))
DF_GAN_CODE_PATH = DF_GAN_PATH / 'code'
SRC_PATH = DF_GAN_CODE_PATH / 'src'
sys.path.insert(0, str(DF_GAN_CODE_PATH))
sys.path.insert(0, str(SRC_PATH))

# Import directly from the DF-GAN repo code
from lib.utils import mkdir_p, load_netG, truncated_noise

class DFGANGenerator:
    """Wrapper class for DF-GAN text-to-image generation."""
    
    def __init__(self, model_path, data_dir, batch_size=1, use_cuda=True, seed=100, steps=50, guidance=7.5):
        self.model_path = model_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
        self.wordtoix = None
        self.text_encoder = None
        self.netG = None
        self.args = self._setup_args()
        
        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        self.args.device = self.device
        self._load_models()
    
    def _setup_args(self):
        """Set up arguments for DF-GAN model."""
        from types import SimpleNamespace
        
        # Basic configuration based on the DF-GAN repo
        args = SimpleNamespace()
        args.z_dim = 100
        args.imsize = 256
        args.cuda = self.use_cuda
        args.manual_seed = self.seed
        args.multi_gpus = False
        args.imgs_per_sent = self.batch_size
        args.train = False
        args.truncation = True
        args.trunc_rate = 0.8
        args.checkpoint = self.model_path
        args.data_dir = self.data_dir
        args.samples_save_dir = str(Path.cwd() / "temp_output")
        
        # Required attributes based on the actual DF-GAN codebase
        args.local_rank = 0
        args.gpu_id = 0
        args.distributed = False
        args.cond_dim = 256
        args.batch_size = self.batch_size
        args.workers = 4
        args.gan_type = 'DFGAN'
        
        # Generator and Discriminator architecture configs from DF-GAN repo
        args.nf = 32  # Number of features
        args.gf_dim = 32
        args.df_dim = 64
        args.ef_dim = 256
        args.n_units = 32
        args.ch_size = 3   # Channel size - required for model initialization
        
        # TEXT namespace as configured in the actual repo
        args.TEXT = SimpleNamespace()
        args.TEXT.WORDS_NUM = 18
        args.TEXT.EMBEDDING_DIM = 256
        args.TEXT.CAPTIONS_PER_IMAGE = 10
        args.TEXT.HIDDEN_DIM = 128
        args.TEXT.RNN_TYPE = 'LSTM'
        
        if "CUB" in str(self.model_path):
            args.dataset = "birds"
            args.n_classes = 10
            args.encoder_epoch = 600
            args.encoder_path = str(DF_GAN_PATH / 'data' / 'birds' / 'DAMSMencoder')
        else:
            args.dataset = "coco"
            args.n_classes = 80
            args.encoder_epoch = 100
            args.encoder_path = str(DF_GAN_PATH / 'data' / 'coco' / 'DAMSMencoder')
            
        # Add additional required params based on repo code
        args.NET_G = ''
        args.NET_D = ''
        args.NET_E = ''
        args.WORKERS = 4  # Uppercase version also used
        args.B_VALIDATION = False
        args.stamp = 'default'
        
        return args
    
    def _load_models(self):
        """Load the DF-GAN models."""
        try:
            # Load word dictionary - Check multiple possible paths
            pickle_paths = [
                os.path.join(self.args.data_dir, f'captions_DAMSM.pickle'),
                os.path.join(self.args.data_dir, f'captions.pickle'),
                # Try with dataset folder name variations
                str(Path(self.args.data_dir).parent / "birds" / "captions_DAMSM.pickle"),
                str(Path(self.args.data_dir).parent / "bird" / "captions_DAMSM.pickle"),
                str(Path(self.args.data_dir).parent / "birds" / "captions.pickle"),
                str(Path(self.args.data_dir).parent / "bird" / "captions.pickle"),
                # Direct hardcoded path that was mentioned in the error
                r"C:\Users\Tejas\Desktop\stl\github\DF-GAN\data\birds\captions_DAMSM.pickle"
            ]
            
            pickle_path = None
            for path in pickle_paths:
                print(f"Checking for pickle at: {path}")
                if os.path.exists(path):
                    pickle_path = path
                    print(f"Found pickle file at: {pickle_path}")
                    break
            
            if pickle_path is None:
                raise FileNotFoundError(f"Cannot find pickle file in any of the expected locations")
            
            # Load the pickle file
            import pickle
            with open(pickle_path, 'rb') as f:
                x = pickle.load(f)
                self.wordtoix = x[3]
                self.args.vocab_size = len(self.wordtoix)
                print(f"Loaded vocabulary with {self.args.vocab_size} words")
                
            # Load models using the exact code structure from the DF-GAN repo
            from lib.perpare import prepare_models
            
            # Check for encoder files and adjust encoder_epoch if needed
            encoder_path = Path(self.args.encoder_path)
            img_encoder = encoder_path / f"image_encoder{self.args.encoder_epoch}.pth"
            text_encoder = encoder_path / f"text_encoder{self.args.encoder_epoch}.pth"
            
            print(f"Loading image encoder from: {img_encoder}")
            print(f"Loading text encoder from: {text_encoder}")
            
            if not img_encoder.exists() or not text_encoder.exists():
                print(f"Warning: {img_encoder} not found. Trying to use image_encoder100.pth instead.")
                print(f"Warning: {text_encoder} not found. Trying to use text_encoder100.pth instead.")
                self.args.encoder_epoch = 100
                
                img_encoder = encoder_path / "image_encoder100.pth"
                text_encoder = encoder_path / "text_encoder100.pth"
                
                print(f"Image encoder path: {img_encoder}")
                print(f"Image encoder exists: {img_encoder.exists()}")
                print(f"Text encoder path: {text_encoder}")
                print(f"Text encoder exists: {text_encoder.exists()}")
            
            _, self.text_encoder, self.netG, _, _ = prepare_models(self.args)
            self.netG = load_netG(self.netG, str(self.model_path), False, train=False)
            self.netG.eval()
            print(f"Models loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def tokenize_text(self, prompt):
        """Convert a prompt to token indices based on the DF-GAN repo implementation"""
        # Convert text to token indices using the loaded wordtoix
        cap_len = len(prompt.split())
        tokens = nltk.tokenize.word_tokenize(prompt.lower())
        tokens = tokens[:18]
        
        cap = []
        for token in tokens:
            if token in self.wordtoix:
                cap.append(self.wordtoix[token])
            else:
                cap.append(0)  # Use 0 for unknown words
        
        # Pad to fixed length
        if len(cap) < 18:
            cap = cap + [0] * (18 - len(cap))
            
        cap = torch.tensor([cap], dtype=torch.long).to(self.device)
        cap_len = torch.tensor([min(cap_len, 18)], dtype=torch.long).to(self.device)
        
        return cap, cap_len
    
    def generate_image(self, prompt, output_dir=None):
        """Generate images based on text prompt."""
        if output_dir is None:
            output_dir = Path(f"temp_output_{int(time.time())}")
        else:
            output_dir = Path(output_dir)
            
        mkdir_p(output_dir)
        
        try:
            # Process the prompt as in the actual DF-GAN code
            cap, cap_len = self.tokenize_text(prompt)
            
            # Get text embeddings using the encoder
            with torch.no_grad():
                hidden = self.text_encoder.init_hidden(1)
                words_embs, sent_emb = self.text_encoder(cap, cap_len, hidden)
                sent_emb = sent_emb.detach()
            
            # Generate noise vector
            if self.args.truncation:
                noise = truncated_noise(self.batch_size, self.args.z_dim, self.args.trunc_rate)
                noise = torch.tensor(noise, dtype=torch.float).to(self.device)
            else:
                noise = torch.randn(self.batch_size, self.args.z_dim).to(self.device)
            
            # Generate image
            with torch.no_grad():
                fake_imgs = self.netG(noise, sent_emb)
                
                # Save image using torchvision
                import torchvision.utils as vutils
                img_path = output_dir / f"generated_{int(time.time())}.png"
                # Use value_range instead of range for newer PyTorch versions
                vutils.save_image(fake_imgs.data, str(img_path), nrow=1, normalize=True, value_range=(-1, 1))
                
            return img_path
            
        except Exception as e:
            print(f"Error generating image: {e}")
            import traceback
            print(traceback.format_exc())
            raise

# Helper functions for direct usage
def setup_generator(model_path, data_dir, use_cuda=True, seed=100):
    """Helper function to set up a generator instance."""
    return DFGANGenerator(model_path, data_dir, use_cuda=use_cuda, seed=seed)

def generate_image(generator, prompt, output_dir=None):
    """Helper function to generate an image with the given generator."""
    return generator.generate_image(prompt, output_dir)
