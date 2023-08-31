import pytorch_lightning as lt
import torch
import torch.nn as nn
from model import EncoderBlock, Encoder, DecoderBlock, Decoder, MultiHeadAttentionBlock, InputEmbeddings, PositionalEncoding, FeedForwardBlock, Transformer
from dataset import casual_mask
import torchmetrics
from model import build_transformer
from TranslatorDataModule import TranslatorDataModule, INPUT_LANGUAGE, OUTPUT_LANGUAGE
from config import get_weights_file_path
from torch.utils.tensorboard import SummaryWriter 
import os

class LtTranslatorModel(lt.LightningModule):
    
    transformer_modules = {
        "encoder": Encoder,
        "encoder_block": EncoderBlock,
        "decoder": Decoder,
        "decoder_block": DecoderBlock,
        "mha_block": MultiHeadAttentionBlock,
        "input_embeddings": InputEmbeddings,
        "pse": PositionalEncoding,
        "ff_block": FeedForwardBlock,
        "Transformer": Transformer
    }

    '''
    src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048
    ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    Configurable Params:
    -------------------
    1. num_epochs: Number of epochs to train the model
    2. d_model: Total dimensions to consider for input embeddings
    3. num_heads: Number of self attention heads
    4. num_layers: Number of Encoder / Decoder Blocks
    5. dropout: Regularization value
    6. d_ff: Expansion Coefficient for First linear layer in FeedForward Module
    7. lang_src: Source Language
    8. lang_tgt: Target Language (For Translation)
    9. lang_dataset: Dataset to be used for Training and Testing
    10. batch_size: Batch Size of Training & Testing
    '''
    def __init__(self, config, datamodule: TranslatorDataModule) -> None:
        super().__init__()
        self.config = config
        self.m_device = config["device"]
        self.datamodule = datamodule

        self.console_width = 80
        
        # Tensorboard 
        self.writer = SummaryWriter(config['experiment_name'])       
        
        try: 
            # get the console window width 
            with os.popen('stty size', 'r') as console: 
                _, console_width = console.read().split() 
                self.console_width = int(console_width) 
        except: 
            # If we can't get the console width, use 80 as default 
            self.console_width = 80 
            
        # Create the directory if it doesn't exist
        save_dir = "weights"
        os.makedirs(save_dir, exist_ok=True)
        
        #Validation variables
        self.val_count = 0 
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 
        self.val_num_examples = 2
        
        #Train variables
        self.train_losses =[] 

    def setup(self, stage: str) -> None:
        lang_in_vocab_size = self.datamodule.get_vocab_size(INPUT_LANGUAGE)
        lang_out_vocab_size = self.datamodule.get_vocab_size(OUTPUT_LANGUAGE)
        lang_in_seq_len = self.datamodule.get_seq_len()
        lang_out_seq_len = self.datamodule.get_seq_len()
        d_model = self.config["d_model"]
        num_layers = self.config["num_layers"]
        num_heads = self.config["num_heads"]
        dropout = self.config["dropout"]
        d_ff = self.config["d_ff"]

        self.model = build_transformer(
                src_vocab_size= lang_in_vocab_size,
                tgt_vocab_size= lang_out_vocab_size,
                src_seq_len= lang_in_seq_len,
                tgt_seq_len= lang_out_seq_len,
                d_model= d_model,
                N = num_layers,
                h = num_heads,
                dropout= dropout,
                d_ff= d_ff
        )

        if self.config['preload']: 
            model_filename = get_weights_file_path(self.config, self.config['preload']) 
            print(f'Preloading model {model_filename}') 
            state = torch.load(model_filename) 
            self.model.load_state_dict(state['model_state_dict'])
            self.trainer.global_step = state['global_step']
            print("Preloaded")
        print("model loaded")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], eps=1e-9)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.datamodule.get_tokenizer(INPUT_LANGUAGE).token_to_id('[PAD]'), label_smoothing=0.1)
        
    
    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device): 
    
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step 
        encoder_output = model.encode(source, source_mask) 

        # Initialize the decoder input with the sos token 
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) 

        while True: 
            if decoder_input.size(1) == max_len:  
                break 

            # build mask for target 
            decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device) 

            # calculate output 
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) 

            # get next token 
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
                ], dim = 1
            )

            if next_word == eos_idx: 
                break 

        return decoder_input.squeeze(0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        device = self.m_device
        encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
        decoder_input = batch['decoder_input'].to(device) # (B, seq_len) 
        encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len) 
        decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len,-seq_len) 
            
        # Run the tensors through the encoder, decoder and the projection layer 
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model) 
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) 
        proj_output = self.model.project(decoder_output) # (B, seq_len, vocab_size) 
            
        # Compare the output with the label 
        label = batch['label'].to(device) # (B, seg_len)
             
        # Compute the loss using a simple cross entropy 
        loss = self.loss_fn(proj_output.view(-1, self.datamodule.get_vocab_size(OUTPUT_LANGUAGE)), label.view(-1)) 
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("loss = ", loss.item(), prog_bar=True) 
        #batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) 
        
        self.train_losses.append(loss.item())         
            
        # Log the loss 
        self.writer.add_scalar('train,loss', loss.item(), self.trainer.global_step) 
        self.writer.flush() 
            
        # Backpropagate the loss 
        loss.backward(retain_graph=True) 
            
        return loss

    def validation_step(self, batch, batch_idx):       
        max_len = self.datamodule.get_seq_len()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        if self.val_count == self.val_num_examples:         
            return 
        
        self.val_count += 1 
        with torch.no_grad():             
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len) 

            # check that the batch size is 1 
            assert encoder_input.size(0) == 1, "Batch  size must be 1 for val"

            model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.datamodule.get_tokenizer(INPUT_LANGUAGE), self.datamodule.get_tokenizer(OUTPUT_LANGUAGE), max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0] 
            model_out_text = self.datamodule.get_tokenizer(OUTPUT_LANGUAGE).decode(model_out.detach().cpu().numpy()) 

            self.val_source_texts.append(source_text) 
            self.val_expected.append(target_text) 
            self.val_predicted.append(model_out_text) 

            # Print the source, target and model output             
            print('-'*self.console_width) 
            print(f"{f'SOURCE: ':>12}{source_text}") 
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")  
            print('-'*self.console_width)
    
    def test_step():
        pass

    def on_validation_epoch_end(self):
        writer = self.writer
        if writer:
            # Evaluate the character error rate 
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate() 
            cer = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('validation cer', cer, self.trainer.global_step) 
            writer.flush() 

            # Compute the word error rate 
            metric = torchmetrics.WordErrorRate() 
            wer = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('validation wer', wer, self.trainer.global_step) 
            writer.flush() 

            # Compute the BLEU metric 
            metric = torchmetrics.BLEUScore() 
            bleu = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('validation BLEU', bleu, self.trainer.global_step) 
            writer.flush() 
            
        self.val_count = 0
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 

    def on_train_epoch_end(self):
        # Save the model at the end of every epoch   
        mean_loss = sum(self.train_losses) / len(self.train_losses)
        print(f'Mean training loss at end of epoch {self.trainer.current_epoch} = {mean_loss}')
        model_filename = get_weights_file_path(self.config, f"{self.trainer.current_epoch:02d}") 
        torch.save({ 
                    'epoch': self.trainer.current_epoch, 
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(), 
                    'global_step': self.trainer.global_step}
                   , model_filename) 
        self.train_losses = []

    def configure_optimizers(self):
        return self.optimizer
    
    
    
