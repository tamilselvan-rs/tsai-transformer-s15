import pytorch_lightning as lt
from datasets import load_dataset
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 
from tokenizers.models import WordLevel
from torch.utils.data import random_split, DataLoader
from dataset import BilingualDataset

INPUT_LANGUAGE = "lang_in"
OUTPUT_LANGUAGE = "lang_out"

class TranslatorDataModule(lt.LightningDataModule):
    TRAIN_SPLIT_PERCENTAGE = 0.9        

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.__meta = {}
        self.__meta[INPUT_LANGUAGE] = {}
        self.__meta[OUTPUT_LANGUAGE] = {}
    
    def __get_or_build_tokenizer(self, config, ds, lang): 
        tokenizer_path = Path(config['tokenizer_file'].format(lang)) 
        if not Path.exists(tokenizer_path): 
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour 
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) 
            tokenizer.pre_tokenizer = Whitespace() 
            trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.__get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path)) 
        else: 
            tokenizer = Tokenizer.from_file(str(tokenizer_path)) 
        return tokenizer 

    def __get_max_sentence_len(self, config, ds, tokenizer_lang_in, tokenizer_lang_out):
        max_len_lang_in = 0 
        max_len_lang_out = 0 

        for item in self.ds_raw: 
            src_ids = self.tokenizer_lang_in.encode(item['translation'][config[INPUT_LANGUAGE]]).ids
            tgt_ids = self.tokenizer_lang_out.encode(item['translation'][config[OUTPUT_LANGUAGE]]).ids
            max_len_lang_in = max(max_len_lang_in, len(src_ids))
            max_len_lang_out = max(max_len_lang_out, len(tgt_ids))
        return (max_len_lang_in, max_len_lang_out)

    def __populate_lang_meta(self, config, ds, tokenizer_lang_in, tokenizer_lang_out):
        # Copy Tokenizer Ref
        self.__meta[INPUT_LANGUAGE]["tokenizer"] = self.tokenizer_lang_in
        self.__meta[OUTPUT_LANGUAGE]["tokenizer"] = self.tokenizer_lang_out

        # Populate Vocab Size
        self.__meta[INPUT_LANGUAGE]["vocab_size"] = self.tokenizer_lang_in.get_vocab_size()
        self.__meta[OUTPUT_LANGUAGE]["vocab_size"] = self.tokenizer_lang_out.get_vocab_size()

        #Populate Max Sentence Length
        max_leng_lang_in, max_len_lang_out = self.__get_max_sentence_len(config, ds, tokenizer_lang_in, tokenizer_lang_out)
        self.__meta[INPUT_LANGUAGE]["max_sentence_len"] = max_leng_lang_in
        self.__meta[OUTPUT_LANGUAGE]["max_sentence_len"] = max_len_lang_out

    def prepare_data(self) -> None:
        config = self.config
        # download
        # It only has the train split, so we divide it overselves 
        self.ds_raw = load_dataset(config['lang_dataset'], f"{config['lang_in']}-{config['lang_out']}", split='train')        
        
        # Build tokenizers 
        self.tokenizer_lang_in = self.__get_or_build_tokenizer(config, self.ds_raw, config['lang_in'])
        self.tokenizer_lang_out = self.__get_or_build_tokenizer(config, self.ds_raw, config['lang_out'])
        
        self.__populate_lang_meta(config, self.ds_raw, self.tokenizer_lang_in, self.tokenizer_lang_out)
    
    def __get_meta(self, src, key):
        if src in self.__meta and key in self.__meta[src]:
            return self.__meta[src][key]

    def get_vocab_size(self, src):
        return self.__get_meta(src, "vocab_size")

    def get_max_sentence_length(self, src):
        return self.__get_meta(src, "max_sentence_len")

    def get_seq_len(self):
        # return max(self.get_max_sentence_length(INPUT_LANGUAGE), self.get_max_sentence_length(OUTPUT_LANGUAGE))
        return 350

    def get_tokenizer(self, src):
        return self.__get_meta(src, "tokenizer")

    def __get_train_val_split(self, ds):
        train_split_size = int(self.TRAIN_SPLIT_PERCENTAGE * len(ds))
        return (train_split_size, len(ds) - train_split_size)

    def setup(self, stage: str) -> None:
        ds = self.ds_raw
        config = self.config
        train_ds_size, val_ds_size = self.__get_train_val_split(ds)
        print(self.get_seq_len())
        print(self.get_vocab_size(INPUT_LANGUAGE))
        print(self.get_vocab_size(OUTPUT_LANGUAGE))
        print(self.get_max_sentence_length(INPUT_LANGUAGE))
        print(self.get_max_sentence_length(OUTPUT_LANGUAGE))

        if stage == "fit":
            train_ds_raw, val_ds_raw = random_split(ds, [train_ds_size, val_ds_size])
            self.train_ds = BilingualDataset(
                                train_ds_raw, 
                                self.tokenizer_lang_in,
                                self.tokenizer_lang_out,
                                config[INPUT_LANGUAGE],
                                config[OUTPUT_LANGUAGE],
                                self.get_seq_len()
                            )
            self.val_ds = BilingualDataset(
                                val_ds_raw,
                                self.tokenizer_lang_in,
                                self.tokenizer_lang_out,
                                config[INPUT_LANGUAGE],
                                config[OUTPUT_LANGUAGE],
                                self.get_seq_len()
                            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True,  num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False,  num_workers=4, persistent_workers=True, pin_memory=True) 
    
    def test_dataloader(self):
        pass